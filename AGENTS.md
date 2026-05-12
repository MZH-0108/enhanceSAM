# Repository Guidelines

## Project Structure & Module Organization
- `models/`: core implementation (`sam_base.py`, `lora_adapter.py`, `boundary_refinement.py`, `enhanced_sam.py`).
- `tests/`: pytest suites mirroring modules (`test_*.py`).
- `configs/train_config.yaml`: baseline training and loss settings.
- `scripts/`, `utils/`, `analysis/`, `experiments/`, `baselines/`, `notebooks/`: workflow directories for training, tooling, and research artifacts (some are scaffolded and filled incrementally).
- `data/`, `checkpoints/`, `logs/`: runtime artifacts, ignored by Git.

## Build, Test, and Development Commands
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Set up a local environment and install dependencies.

```bash
pytest tests -v
pytest tests -v --cov=models --cov=utils
```
Run unit tests and coverage checks.

```bash
black models tests
isort models tests
flake8 models tests
```
Format and lint before opening a PR.

## Coding Style & Naming Conventions
- Target Python 3.8+, PEP 8, 4-space indentation.
- Use type hints for public APIs and clear docstrings on modules/classes/functions.
- Naming: modules/files `snake_case`, classes `PascalCase`, functions/variables `snake_case`, constants `UPPER_SNAKE_CASE`.
- Keep tensor contracts explicit (shape expectations and key names such as `masks`, `iou_pred`, `refined_mask`).
- **注释规范（强制）**：新增或修改的核心代码必须提供详细中文注释，至少说明“这段代码做什么、输入输出是什么、为什么这样做”；复杂张量变换需标注 shape 变化。

## Testing Guidelines
- Framework: `pytest` with `pytest-cov`.
- Add tests in `tests/test_<module>.py` for each changed module.
- Prefer mock-based or synthetic tensor tests; avoid external downloads in unit tests.
- For numerical code, assert shapes plus NaN/Inf safety.
- Aim for strong coverage on touched paths (use `--cov` output to validate).

## Commit & Pull Request Guidelines
- Follow the existing commit pattern: `type(scope): subject` (for example, `feat(models): add boundary loss weighting`).
- Keep commits small and focused; run tests before committing.
- PRs should include: purpose, affected files/modules, test command/results, and config deltas (for example `configs/train_config.yaml`).
- If model behavior changes, include representative metric or qualitative output notes in the PR description.

## Execution Focus Rules
- Before each coding task, read `PROJECT_STATE.md` and only execute the listed “Current In Progress” item to avoid scope drift.
- After each task, update `PROJECT_STATE.md` with completed work, next 1-3 steps, and artifact paths.
- If requirements change mid-task, update the “Frozen Decisions” section in `PROJECT_STATE.md` first, then implement.
