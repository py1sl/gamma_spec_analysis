"""Pulse-height spectrum data model with type annotations."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence, Union, Any, Dict
import numpy as np
import numpy.typing as npt

# Minimum supported number of energy-fit coefficients (intercept + slope).
_REQUIRED_EFIT_LEN = 2


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

    def generate_ebins(self) -> npt.NDArray[np.float64]:
        """Generate energy bin boundaries from the energy fit coefficients.

        Uses the linear calibration ``E = a0 + a1 * channel`` stored in
        :attr:`energy_fit_coefficients` to produce one energy value per
        channel.  If :attr:`num_channels` is zero it is derived from
        :attr:`counts` and updated in place.

        Returns
        -------
        numpy.ndarray of float64
            Energy values for each channel, length equal to
            :attr:`num_channels`.

        Raises
        ------
        ValueError
            If :attr:`energy_fit_coefficients` is ``None`` or does not
            contain exactly two coefficients.
        """
        if self.energy_fit_coefficients is None:
            raise ValueError(
                "energy_fit_coefficients is not set; cannot generate energy bins"
            )

        # Ensure num_channels is set
        if self.num_channels == 0:
            self.num_channels = int(self.counts.size)

        energy_coeffs = self.energy_fit_coefficients
        if len(energy_coeffs) != _REQUIRED_EFIT_LEN:
            raise ValueError(
                "energy_fit_coefficients must contain exactly 2 elements"
            )

        x = np.arange(self.num_channels)
        return (energy_coeffs[0] + x * energy_coeffs[1]).astype(np.float64)

    def __add__(self, other: "PhSpectrum") -> "PhSpectrum":
        """Return a new spectrum whose counts are the element-wise sum of *self*
        and *other*.

        Rules applied to the metadata of the result:

        * **counts** – element-wise integer sum.
        * **live_time** / **real_time** – summed when both operands carry the
          value; ``None`` when either operand is missing it.
        * **start_chan_num** / **energy_fit_coefficients** /
          **efficiency_fit_coefficients** / **spec_name** / **start_time** –
          taken from *self* (the left operand).
        * **channels** – regenerated as ``range(num_channels)``.
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
            start_chan_num=self.start_chan_num,
            channels=list(range(int(summed_counts.size))),
            counts=summed_counts,
            live_time=live,
            real_time=real,
            energy_fit_coefficients=self.energy_fit_coefficients,
            efficiency_fit_coefficients=self.efficiency_fit_coefficients,
            start_time=self.start_time,
        )

    def __sub__(self, other: "PhSpectrum") -> "PhSpectrum":
        """Return a new spectrum whose counts are the element-wise difference
        of *self* and *other*.

        Background subtraction is the primary use-case: the left operand is
        the sample spectrum and the right operand is the (appropriately
        time-normalised) background.  Because statistical fluctuations can
        produce negative bin values the result counts are stored as **int64**
        (signed), preserving negative values rather than clamping them.

        Rules applied to the metadata of the result:

        * **counts** – element-wise integer difference (``self - other``).
        * **live_time** / **real_time** – taken from *self* (the sample).
        * **start_chan_num** / **energy_fit_coefficients** /
          **efficiency_fit_coefficients** / **spec_name** / **start_time** –
          taken from *self* (the left operand).
        * **channels** – regenerated as ``range(num_channels)``.
        * All other fields are left at their defaults.

        Parameters
        ----------
        other:
            Another :class:`PhSpectrum` to subtract from this one.

        Returns
        -------
        PhSpectrum
            A new spectrum with subtracted counts.

        Raises
        ------
        ValueError
            If the two spectra have different numbers of channels.
        """
        if not isinstance(other, PhSpectrum):
            return NotImplemented

        if self.num_channels != other.num_channels:
            raise ValueError(
                f"Cannot subtract spectra with different channel counts: "
                f"{self.num_channels} vs {other.num_channels}"
            )

        diff_counts = (
            np.asarray(self.counts, dtype=np.int64)
            - np.asarray(other.counts, dtype=np.int64)
        )

        return PhSpectrum(
            spec_name=self.spec_name,
            start_chan_num=self.start_chan_num,
            channels=list(range(int(diff_counts.size))),
            counts=diff_counts,
            live_time=self.live_time,
            real_time=self.real_time,
            energy_fit_coefficients=self.energy_fit_coefficients,
            efficiency_fit_coefficients=self.efficiency_fit_coefficients,
            start_time=self.start_time,
        )

    def normalise_to_livetime(self) -> npt.NDArray[np.float64]:
        """Return counts normalised by live time (counts per second).

        Each bin value is divided by :attr:`live_time`, producing an array of
        count rates suitable for comparing spectra acquired over different
        measurement durations.

        Returns
        -------
        numpy.ndarray of float64
            Count rate in each channel (counts / second).

        Raises
        ------
        ValueError
            If :attr:`live_time` is ``None`` or zero.
        """
        if self.live_time is None:
            raise ValueError("live_time is not set; cannot normalise to live time")
        if self.live_time == 0.0:
            raise ValueError("live_time is zero; cannot normalise to live time")
        return self.counts.astype(np.float64) / self.live_time

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict (converts ndarrays to lists)."""
        d = asdict(self)
        # Convert numpy arrays to lists for serialization
        if isinstance(self.counts, np.ndarray):
            d["counts"] = self.counts.tolist()
        if isinstance(self.ebin, np.ndarray):
            d["ebin"] = self.ebin.tolist()
        return d
