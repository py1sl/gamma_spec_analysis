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
        self.ebin = (
            np.asarray(self.ebin, dtype=np.float64)
            if self.ebin
            else np.array([], dtype=np.float64)
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
