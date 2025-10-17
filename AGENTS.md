# Repository Guidelines

## Project Structure & Module Organization
Core training entry points live in `app/`, with `main.py` for local jobs and `main_distributed.py` for Slurm-style launches; action-conditioned logic sits in `app/vjepa_droid`. Experiment settings are versioned under `configs/` (`train`, `eval`, `inference`), so add new YAMLs there instead of hard-coding parameters. Reusable components live in `src/` (`datasets`, `models`, `masks`, `utils`), while evaluation loops sit in `evals/` with matching local and distributed runners. Tests accompany major modules in `tests/`, assets sit in `assets/`, and exploratory notebooks stay in `notebooks/`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create an isolated environment before installing deps.
- `pip install -r requirements.txt` plus `requirements-test.txt` when you need GPU-enabled pytest extras.
- `python -m app.main --fname configs/train/vitl16/pretrain-256px-16f.yaml --devices cuda:0`: launch a local pretraining run; swap the YAML to target other model sizes or phases.
- `python -m evals.main --fname configs/eval/vitl16/ssv2.yaml --devices cuda:0 cuda:1`: run the attentive-probe evaluation stack; use `evals.main_distributed` for cluster jobs.
- `pytest tests`: execute the repository test suite; narrow to `pytest tests/models/test_vision_transformer.py` during GPU debugging.

## Coding Style & Naming Conventions
Use 4-space indentation and keep lines ≤119 characters, matching `pyproject.toml` and `CONTRIBUTING.md`. Python modules, configs, and directories follow snake_case, while classes remain PascalCase and configuration tags use hyphenated slugs (e.g., `pretrain-256px-16f`). Format with `black`, organize imports with `isort --profile black`, and run `flake8` before sending a PR. Maintain config keys and CLI flags in lowercase to align with existing YAML patterns.

## Testing Guidelines
Pytest is the supported framework, with GPU-heavy suites marked via `pytest.mark.skipif` when CUDA is unavailable. Add targeted tests alongside the code they exercise within `tests/`, mirroring the source layout. For new training or evaluation flows, include a smoke test that covers config parsing and scheduler wiring; document hardware needs in the test docstring. Keep long-running integration checks optional behind an environment flag.

## Commit & Pull Request Guidelines
Follow the existing Git history by writing imperative, single-sentence commit subjects and appending the GitHub PR or issue number (e.g., `Add decord instructions (#82)`). Branch from `main`, keep commits focused, and update related configs or docs when behavior changes. Pull requests must note the motivation, list runnable commands, and link tracked issues; include screenshots or logs for user-facing updates. Ensure `pytest` and linters pass before requesting review, sign the Meta CLA once, and add reviewers familiar with the touched subsystems.
