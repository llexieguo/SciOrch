"""SciOrch package."""

__all__ = ["OrchestratorConfig"]


def __getattr__(name: str):
    if name == "OrchestratorConfig":
        from sciorch.config import OrchestratorConfig

        return OrchestratorConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
