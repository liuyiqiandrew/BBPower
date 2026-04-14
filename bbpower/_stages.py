from __future__ import annotations

from importlib import import_module
from typing import Any


STAGE_MODULES: dict[str, str] = {
    "BBPowerSpecter": "bbpower.power_specter",
    "BBPowerSummarizer": "bbpower.power_summarizer",
    "BBCompSep": "bbpower.compsep",
    "BBPlotter": "bbpower.plotter",
}


def get_stage_class(stage_name: str) -> Any:
    try:
        module_name = STAGE_MODULES[stage_name]
    except KeyError as exc:
        known = ", ".join(sorted(STAGE_MODULES))
        raise KeyError(f"Unknown BBPower stage {stage_name!r}. Known stages: {known}") from exc

    module = import_module(module_name)
    return getattr(module, stage_name)
