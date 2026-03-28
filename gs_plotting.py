# -*- coding: utf-8 -*-
"""
gamma spectrum plotting functions
"""

import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Sequence, Union, Any
import numpy.typing as npt


def plot_spect_peaks(
    smooth_counts: npt.NDArray[Any],
    ebins: npt.NDArray[Any],
    peaks: Sequence[int],
    fname: Optional[str] = None,
) -> None:
    """Plots the spectra and highlights the peaks on the plot"""
    plt.clf()
    for peak in peaks:
        plt.plot(ebins[peak], smooth_counts[peak], "xr")
    plt.plot(ebins, smooth_counts)
    plt.xlabel("ebins")
    plt.ylabel("counts")
    plt.yscale("log")

    if fname:
        plt.savefig(fname)
    else:
        plt.show()


def plot_spec(
    counts: Union[Sequence[int], npt.NDArray[Any]],
    erg: Optional[npt.NDArray[Any]] = None,
    fname: Optional[str] = None,
) -> None:
    """simple plotting routine for spectra"""
    counts = np.array(counts).astype(int)
    plt.clf()

    if erg is None:
        x = np.arange(len(counts))
    else:
        x = erg

    plt.yscale("log")
    plt.step(x, counts)

    if fname:
        plt.savefig(fname)
    else:
        plt.show()


if __name__ == "__main__":
    pass
