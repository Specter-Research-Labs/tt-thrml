from __future__ import annotations


class HostFallbackError(RuntimeError):
    """Raised when a TT hot-path operation would silently execute on the host."""


def raise_host_fallback_disabled(operation: str, *, remedy: str | None = None) -> None:
    message = (
        f"Host fallback disabled for {operation}. This path would synchronize TT "
        "device tensors to CPU in the sampler hot path."
    )
    if remedy:
        message = f"{message} {remedy}"
    raise HostFallbackError(message)
