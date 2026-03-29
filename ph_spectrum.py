"""Pulse-height spectrum data model with type annotations."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence, Union, Any, Dict
import numpy as np
import numpy.typing as npt


@dataclass
class PhSpectrum:
    spec_name: str = ""
    start_chan_num: int = 0
    num_channels: int = 0
    channels: List[int] = field(default_factory=list)
    ebin: Union[List[float], npt.NDArray[np.float64]] = field(default_factory=list)
    counts: Union[List[int], npt.NDArray[np.int64]] = field(default_factory=list)
    live_time: Optional[float] = None
    real_time: Optional[float] = None
    file_path: str = ""
    start_time: Optional[str] = None
    peaks: List[int] = field(default_factory=list)
    energy_fit_coefficients: Optional[Sequence[float]] = None
    efficiency_fit_coefficients: Optional[Sequence[float]] = None
    keywords: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize common inputs to numpy arrays and set derived fields."""
        # Ensure counts is a numpy array of integer type for numeric ops
        self.counts = np.asarray(self.counts, dtype=np.int64)

        # Set num_channels if not provided and counts present
        if self.num_channels == 0 and self.counts.size > 0:
            self.num_channels = int(self.counts.size)

        # Normalize ebin to numpy array (use float dtype)
        ebin_input = self.ebin if self.ebin is not None else []
        self.ebin = np.asarray(ebin_input, dtype=np.float64)

    def __add__(self, other: "PhSpectrum") -> "PhSpectrum":
        """Return a new spectrum whose counts are the element-wise sum of *self*
        and *other*.

        Rules applied to the metadata of the result:

        * **counts** – element-wise integer sum.
        * **live_time** / **real_time** – summed when both operands carry the
          value; ``None`` when either operand is missing it.
        * **energy_fit_coefficients** / **efficiency_fit_coefficients** /
          **spec_name** / **start_time** – taken from *self* (the left operand).
        * All other fields are left at their defaults.

        Parameters
        ----------
        other:
            Another :class:`PhSpectrum` to add to this one.

        Returns
        -------
        PhSpectrum
            A new spectrum with summed counts.

        Raises
        ------
        ValueError
            If the two spectra have different numbers of channels.
        """
        if not isinstance(other, PhSpectrum):
            return NotImplemented

        if self.num_channels != other.num_channels:
            raise ValueError(
                f"Cannot add spectra with different channel counts: "
                f"{self.num_channels} vs {other.num_channels}"
            )

        summed_counts = (
            np.asarray(self.counts, dtype=np.int64)
            + np.asarray(other.counts, dtype=np.int64)
        )

        live = (
            self.live_time + other.live_time
            if self.live_time is not None and other.live_time is not None
            else None
        )
        real = (
            self.real_time + other.real_time
            if self.real_time is not None and other.real_time is not None
            else None
        )

        return PhSpectrum(
            spec_name=self.spec_name,
            counts=summed_counts,
            live_time=live,
            real_time=real,
            energy_fit_coefficients=self.energy_fit_coefficients,
            efficiency_fit_coefficients=self.efficiency_fit_coefficients,
            start_time=self.start_time,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict (converts ndarrays to lists)."""
        d = asdict(self)
        # Convert numpy arrays to lists for serialization
        if isinstance(self.counts, np.ndarray):
            d["counts"] = self.counts.tolist()
        if isinstance(self.ebin, np.ndarray):
            d["ebin"] = self.ebin.tolist()
        return d
