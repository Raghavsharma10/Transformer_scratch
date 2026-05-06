def set_signal_type(self, sig_type):
        """Set the signal type of interest.

        Sets the signal type for which the SNR is calculated.
        This means inspiral, merger, and/or ringdown.

        Args:
            sig_type (str or list of str): Signal type desired by user.
                Choices are `ins`, `mrg`, `rd`, `all` for circular waveforms created with PhenomD.
                If eccentric waveforms are used, must be `all`.

        """
        if isinstance(sig_type, str):
            sig_type = [sig_type]
        self.snr_input.signal_type = sig_type
        return