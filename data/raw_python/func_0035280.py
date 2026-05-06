def set_wd_noise(self, wd_noise):
        """Add White Dwarf Background Noise

        This adds the White Dwarf (WD) Background noise. This can either do calculations with,
        without, or with and without WD noise.

        Args:
            wd_noise (bool or str, optional): Add or remove WD background noise. First option is to
                have only calculations with the wd_noise. For this, use `yes` or True.
                Second option is no WD noise. For this, use `no` or False. For both calculations
                with and without WD noise, use `both`.

        Raises:
            ValueError: Input value is not one of the options.

        """
        if isinstance(wd_noise, bool):
            wd_noise = str(wd_noise)

        if wd_noise.lower() == 'yes' or wd_noise.lower() == 'true':
            wd_noise = 'True'
        elif wd_noise.lower() == 'no' or wd_noise.lower() == 'false':
            wd_noise = 'False'
        elif wd_noise.lower() == 'both':
            wd_noise = 'Both'
        else:
            raise ValueError('wd_noise must be yes, no, True, False, or Both.')

        self.sensitivity_input.add_wd_noise = wd_noise
        return