# Emotion Simulation Submission Package

This directory contains the core simulation implementation, executable prompt templates, versioned experimental stimuli, and the complete reported experiment outputs.

## Contents

- `formulas.py`: exposure, emotion-update, feedback, and group-level indicator equations.
- `models.py`: content, agent-profile, perception, action, and scenario data structures.
- `platform_mechanisms.py`: network construction and feed-exposure selection.
- `agent.py`: agent perception, emotion updating, and behavioral decisions.
- `simulation.py`: simulation orchestration and result aggregation.
- `prompts.py`, `generation.py`, `scoring.py`, `llm_client.py`: LLM prompt and structured-output pipeline.
- `experiments.py`, `plotting.py`, `main.py`: experiment orchestration and figure generation.
- `data/experimental_stimuli.json`: versioned, read-only study stimuli and item-level scores.
- `outputs/experiments/`: run-level and aggregate outputs for the reported experiments.
- `outputs/figures/`: figures generated from the reported experiment outputs.

## Configuration

Copy `.env.example` to `.env` only when a new local run is required, then add credentials locally. The submission package contains no credentials.

The published tables and figures should be checked against the archived run-level files in `outputs/experiments/`; no new LLM calls are required for inspecting the reported results.

## Data status

The experimental stimuli are research inputs. They are read-only, versioned, and are never updated by the simulation. Runtime comments, replies, perceptions, and decisions are generated when a new simulation is explicitly launched.
