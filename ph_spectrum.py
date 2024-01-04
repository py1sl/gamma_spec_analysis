""" """


class PhSpectrum(object):
    """Pulse height spectrum class"""

    def __init__(
        self,
        spec_name="",
        start_chan_num=0,
        num_channels=0,
        channels=None,
        ebin=None,
        counts=None,
        live_time=None,
        real_time=None,
        file_path="",
        start_time=None,
        peaks=None,
        efit_co_eff=None,
        eff_fit_co_eff=None,
    ):
        """Initialise Pulse height Spectrum variables"""
        self.channels = [] if channels is None else channels
        self.ebin = [] if ebin is None else ebin
        self.counts = [] if counts is None else counts
        self.spec_name = spec_name
        self.start_chan_num = start_chan_num
        self.num_channels = num_channels
        self.real_time = real_time
        self.live_time = live_time
        self.file_path = file_path
        self.start_time = start_time
        self.peaks = [] if peaks is None else peaks
        self.efit_co_eff = efit_co_eff
        self.eff_fit_co_eff = eff_fit_co_eff
