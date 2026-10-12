from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from threading import Lock
from time import sleep
from typing import Any, Optional

try:
    import requests
except Exception as import_error:
    requests = None
    REQUESTS_IMPORT_ERROR = import_error
else:
    REQUESTS_IMPORT_ERROR = None

from prompts import STRUCTURED_JSON_SYSTEM_PROMPT


def _extract_json_text(text: str) -> str:
    """从模型返回内容中提取 JSON 文本。"""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    start_list = cleaned.find("[")
    start_obj = cleaned.find("{")
    starts = [value for value in [start_list, start_obj] if value >= 0]
    if not starts:
        return cleaned
    return cleaned[min(starts) :]


@dataclass
class LLMClient:
    """OpenAI 兼容风格 HTTP 调用器。

    该客户端可同时服务材料生成阶段和代理运行时阶段。
    """

    api_key: Optional[str]
    base_url: str
    model: str
    timeout: int = 60
    request_count: int = field(default=0, init=False)
    failure_count: int = field(default=0, init=False)
    batch_request_count: int = field(default=0, init=False)
    inference_item_count: int = field(default=0, init=False)
    valid_inference_item_count: int = field(default=0, init=False)
    invalid_inference_item_count: int = field(default=0, init=False)
    fallback_item_count: int = field(default=0, init=False)
    input_token_count: int = field(default=0, init=False)
    output_token_count: int = field(default=0, init=False)
    _counter_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def _record_request(self) -> None:
        with self._counter_lock:
            self.request_count += 1

    def _record_failure(self) -> None:
        with self._counter_lock:
            self.failure_count += 1

    def record_invalid_response(self) -> None:
        """登记调用成功但不符合当前结构化任务约束的响应。"""

        self._record_failure()

    def _record_batch_result(self, requested: int, valid: int) -> None:
        """记录批量请求覆盖的 Agent 条目及有效结果。"""

        with self._counter_lock:
            self.batch_request_count += 1
            self.inference_item_count += max(0, int(requested))
            self.valid_inference_item_count += max(0, int(valid))
            self.invalid_inference_item_count += max(0, int(requested) - int(valid))

    def record_fallback_items(self, count: int) -> None:
        """记录最终回退到规则逻辑的 Agent 条目数。"""

        with self._counter_lock:
            self.fallback_item_count += max(0, int(count))

    def _record_usage(self, data: dict[str, Any]) -> None:
        """Record token usage from Chat Completions or Responses payloads."""

        usage = data.get("usage")
        if not isinstance(usage, dict):
            return
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
        try:
            normalized_input = max(0, int(input_tokens or 0))
            normalized_output = max(0, int(output_tokens or 0))
        except (TypeError, ValueError):
            return
        with self._counter_lock:
            self.input_token_count += normalized_input
            self.output_token_count += normalized_output

    def estimated_cost_rmb(
        self,
        input_price_per_million: float = 1.0,
        output_price_per_million: float = 2.0,
    ) -> float:
        """Estimate runtime cost using the caller's current token tariff."""

        return (
            self.input_token_count * input_price_per_million
            + self.output_token_count * output_price_per_million
        ) / 1_000_000.0

    def has_api_key(self) -> bool:
        """判断是否具备真实调用条件。"""

        return bool(self.api_key and self.api_key.strip())

    def _uses_responses_api(self) -> bool:
        """Detect an OpenAI Responses endpoint from the configured URL."""

        return self.base_url.rstrip("/").endswith("/responses")

    def _endpoint_url(self) -> str:
        """Build the configured OpenAI-compatible endpoint without double suffixes."""

        normalized = self.base_url.rstrip("/")
        if normalized.endswith("/chat/completions") or normalized.endswith("/responses"):
            return normalized
        suffix = "/responses" if self._uses_responses_api() else "/chat/completions"
        return f"{normalized}{suffix}"

    def _request_payload(
        self,
        prompt: str,
        temperature: float,
        max_output_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Use the request shape required by the selected OpenAI-compatible API."""

        system_prompt = STRUCTURED_JSON_SYSTEM_PROMPT
        if self._uses_responses_api():
            payload = {
                "model": self.model,
                "instructions": system_prompt,
                "input": prompt,
                "temperature": temperature,
            }
            if max_output_tokens is not None:
                payload["max_output_tokens"] = max(1, int(max_output_tokens))
            return payload
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        if "api.deepseek.com" in self.base_url.lower():
            payload["thinking"] = {"type": "disabled"}
        if max_output_tokens is not None:
            payload["max_tokens"] = max(1, int(max_output_tokens))
        return payload

    def _extract_response_content(self, data: dict[str, Any]) -> str:
        """Extract text from either Chat Completions or Responses payloads."""

        if not self._uses_responses_api():
            return str(data["choices"][0]["message"]["content"]).strip()

        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        for output_item in data.get("output", []):
            if not isinstance(output_item, dict):
                continue
            for content_item in output_item.get("content", []):
                if isinstance(content_item, str) and content_item.strip():
                    return content_item.strip()
                if not isinstance(content_item, dict):
                    continue
                text = content_item.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
        return ""

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """请求文本结果。失败时返回 `None`。"""

        if not self.has_api_key():
            self._record_failure()
            return None
        if requests is None:
            self._record_failure()
            print(f"[LLM] requests 不可用，将回退到启发式逻辑。错误信息：{REQUESTS_IMPORT_ERROR}")
            return None

        payload = self._request_payload(prompt, temperature, max_output_tokens)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        endpoint = self._endpoint_url()
        retryable_statuses = {429, 500, 502, 503, 504}
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                self._record_request()
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code in retryable_statuses:
                    last_error = requests.HTTPError(response.text, response=response)
                    if attempt < 2:
                        sleep(1.5 * (attempt + 1))
                        continue
                elif response.status_code == 401:
                    # GLM 偶发返回 1001 时重试一次，避免网关瞬时鉴权异常直接触发规则回退。
                    try:
                        error_code = str(response.json().get("error", {}).get("code", ""))
                    except ValueError:
                        error_code = ""
                    if error_code == "1001" and attempt < 1:
                        last_error = requests.HTTPError(response.text, response=response)
                        sleep(1.0)
                        continue
                    last_error = requests.HTTPError(response.text, response=response)
                    break
                response.raise_for_status()
                data = response.json()
                self._record_usage(data)
                content = self._extract_response_content(data)
                if not content:
                    self._record_failure()
                    return None
                return content
            except requests.RequestException as error:
                last_error = error
                if attempt < 2:
                    sleep(1.5 * (attempt + 1))
                    continue
            except (KeyError, IndexError, TypeError, ValueError) as error:
                self._record_failure()
                print(f"[LLM] 返回结果无法解析，将回退到启发式逻辑。错误信息：{error}")
                return None

        self._record_failure()
        print(
            "[LLM] 请求失败，将回退到启发式逻辑。"
            f"model={self.model} | endpoint={endpoint} | Authorization=已配置 | 错误信息：{last_error}"
        )
        return None

    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_output_tokens: Optional[int] = None,
    ) -> Optional[Any]:
        """请求并解析 JSON。"""

        text = self.generate_text(
            prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        if not text:
            return None
        try:
            return json.loads(_extract_json_text(text))
        except json.JSONDecodeError:
            self._record_failure()
            return None

    def generate_json_batch(
        self,
        prompt: str,
        expected_ids: list[int],
        temperature: float = 0.3,
    ) -> dict[int, Any]:
        """请求批量 Agent 结果，并按输入 ID 接受其中有效的独立条目。"""

        normalized_ids = [int(agent_id) for agent_id in expected_ids]
        expected = set(normalized_ids)
        max_output_tokens = min(6144, max(4096, 96 * len(normalized_ids) + 2048))
        payload = self.generate_json(
            prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        rows: Any = payload.get("results") if isinstance(payload, dict) else payload
        results: dict[int, Any] = {}

        if isinstance(rows, list):
            for row in rows:
                try:
                    if isinstance(row, list) and row:
                        agent_id = int(row[0])
                    elif isinstance(row, dict):
                        agent_id = int(row.get("agent_id"))
                    else:
                        continue
                except (TypeError, ValueError, IndexError):
                    continue
                if agent_id not in expected or agent_id in results:
                    continue
                results[agent_id] = row

        self._record_batch_result(len(normalized_ids), len(results))
        if len(results) < len(expected):
            self.record_invalid_response()
        return results
