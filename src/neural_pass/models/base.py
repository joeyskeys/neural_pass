"""StyleModel protocol / ABC: load, generate, fine_tune.

A style config selects which customized (post-trained) model to use.
Implementations belong in future modules; this file only defines the frame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Protocol, runtime_checkable

from neural_pass.io.style_config import StyleConfig


@runtime_checkable
class StyleModel(Protocol):
    """Protocol for style generative models used by the pass."""

    def load(self) -> None:
        """Load base or fine-tuned weights."""

    def generate(self, conditioning: Mapping[str, Any]) -> Mapping[str, Any]:
        """Produce a stylized image from conditioning (+ style prompt)."""

    def fine_tune(self, train_spec: Mapping[str, Any]) -> Mapping[str, Any]:
        """Post-train / fine-tune into a customized style model."""


class StyleModelABC(ABC):
    """Optional ABC mirroring the StyleModel protocol."""

    def __init__(self, style: StyleConfig) -> None:
        self.style = style

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def generate(self, conditioning: Mapping[str, Any]) -> Mapping[str, Any]: ...

    @abstractmethod
    def fine_tune(self, train_spec: Mapping[str, Any]) -> Mapping[str, Any]: ...


class PlaceholderStyleModel(StyleModelABC):
    """Scaffold implementation — clearly stubbed, no fake metrics."""

    def load(self) -> None:
        print(
            f"    [StyleModel] load base={self.style.base_model_id!r} "
            f"style_ckpt={self.style.style_model_path!r} (stub)"
        )
        # TODO: load real checkpoint.

    def generate(self, conditioning: Mapping[str, Any]) -> Mapping[str, Any]:
        print("    [StyleModel] generate (stub)")
        return {
            "status": "stub",
            "conditioning_keys": list(conditioning.keys()),
            "image": None,
            "note": "PlaceholderStyleModel.generate — not implemented.",
        }

    def fine_tune(self, train_spec: Mapping[str, Any]) -> Mapping[str, Any]:
        print("    [StyleModel] fine_tune (stub)")
        return {
            "status": "stub",
            "train_spec_keys": list(train_spec.keys()),
            "note": "PlaceholderStyleModel.fine_tune — not implemented.",
        }
