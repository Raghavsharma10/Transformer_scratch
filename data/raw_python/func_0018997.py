def prepare_allseries(self, ramflag: bool = True) -> None:
        """Prepare the |IOSequence.series| objects of all `input`, `flux` and
        `state` sequences of the model handled by this element.

        Call this method before a simulation run, if you need access to
        (nearly) all simulated series of the handled model after the
        simulation run is finished.

        By default, the time series are stored in RAM, which is the faster
        option.  If your RAM is limited, pass |False| to function argument
        `ramflag` to store the series on disk.
        """
        self.prepare_inputseries(ramflag)
        self.prepare_fluxseries(ramflag)
        self.prepare_stateseries(ramflag)