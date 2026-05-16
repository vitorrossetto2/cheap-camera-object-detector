# AGENTS.md

## Project Overview

Image Behaviour Alerts is a Python project for working with camera streams. It can:

- capture images from RTSP streams;
- monitor a camera source with Ultralytics YOLO object detection;
- send alerts when a configured target object is detected.

The main entry point is `main.py`, which starts the desktop UI. Core
implementation lives under `src/image_behaviour_alerts/`, with focused modules
for detection, monitoring, notifications, and RTSP capture. Tests live in
`tests/`.

The project uses `uv` for dependency and command execution. Prefer commands such
as:

```powershell
uv run python main.py
uv run python -m unittest discover -s tests
```

## Development Directives

- Keep UI startup centered in `main.py` and reusable logic in
  `src/image_behaviour_alerts/`.
- Prefer small, testable functions and classes. Add or update unit tests when
  changing detection matching, alert behavior, monitoring flow, or other
  reusable business logic.
- Do not commit local secrets. Runtime camera values should come from `.env`;
  document required variables in `.env.example`.
- Preserve existing command names and arguments unless the change explicitly
  requires a breaking CLI update.
- When adding dependencies, update `pyproject.toml` and the lockfile through
  `uv`.

## VS Code Debugger Rule

Whenever you create a new feature that can be called in a different way, update
`.vscode/launch.json` in the same change so new developers can debug it easily.

This applies when adding or changing any callable surface, including:

- a new `main.py` subcommand;
- a new mode or meaningful option for an existing subcommand;
- a new script, module entry point, or operational workflow;
- a feature that can be run against a different source, target, model, output,
  or runtime configuration.

Debugger updates should include a practical launch configuration with sensible
defaults and any required `inputs` entries. Prefer short, descriptive names such
as `UI: desktop (.env)`. Use `${workspaceFolder}` for paths,
`${workspaceFolder}/.env` for environment loading when needed, and values that
are safe for local debugging.

If a feature is intentionally not suitable for VS Code debugging, document the
reason in the pull request or change notes.

## Verification

Before finishing a change, run the most relevant checks available for the scope:

```powershell
uv run python -m unittest discover -s tests
```

For UI entry-point changes, run the app path when practical:

```powershell
uv run python main.py
```

For typing changes, run:

```powershell
uv run pyright
```
