"""Configuration objects for telemetry runtime."""

from dataclasses import asdict, dataclass
import os


def _parse_bool(value: str, *, field_name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean for {field_name}: {value!r}")


@dataclass(slots=True)
class TelemetryConfig:
    """Runtime configuration for MLflow telemetry integration."""

    enabled: bool = True
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str | None = None
    sampling_rate: float = 1.0
    log_content: bool = True
    fail_open: bool = True
    service_name: str = "mlflow-features"
    service_version: str | None = None
    environment: str = "dev"

    def __post_init__(self) -> None:
        if not 0.0 <= self.sampling_rate <= 1.0:
            raise ValueError("sampling_rate must be between 0.0 and 1.0")
        if not self.service_name.strip():
            raise ValueError("service_name must be non-empty")
        if not self.environment.strip():
            raise ValueError("environment must be non-empty")

    @classmethod
    def from_env(cls, prefix: str = "AGENT_TELEMETRY_") -> "TelemetryConfig":
        """Create config from environment variables with sensible defaults."""

        value = os.getenv
        kwargs: dict[str, object] = {}
        if (raw := value(f"{prefix}ENABLED")) is not None:
            kwargs["enabled"] = _parse_bool(raw, field_name=f"{prefix}ENABLED")
        if (raw := value(f"{prefix}MLFLOW_TRACKING_URI")):
            kwargs["mlflow_tracking_uri"] = raw
        if (raw := value(f"{prefix}MLFLOW_EXPERIMENT")):
            kwargs["mlflow_experiment"] = raw
        if (raw := value(f"{prefix}SAMPLING_RATE")) is not None:
            kwargs["sampling_rate"] = float(raw)
        if (raw := value(f"{prefix}LOG_CONTENT")) is not None:
            kwargs["log_content"] = _parse_bool(raw, field_name=f"{prefix}LOG_CONTENT")
        if (raw := value(f"{prefix}FAIL_OPEN")) is not None:
            kwargs["fail_open"] = _parse_bool(raw, field_name=f"{prefix}FAIL_OPEN")
        if (raw := value(f"{prefix}SERVICE_NAME")):
            kwargs["service_name"] = raw
        if (raw := value(f"{prefix}SERVICE_VERSION")):
            kwargs["service_version"] = raw
        if (raw := value(f"{prefix}ENVIRONMENT")):
            kwargs["environment"] = raw
        return cls(**kwargs)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot for downstream sinks."""
        return asdict(self)
