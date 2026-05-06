def _set_scalers(self):
        """
        Set the variables self._scalers as given by self.scalers,
        if self.scalers is None, then a default value is used.
        """

        # Set default value for rep_scalers if None
        if self.rep_scalers is None:
            # Draw self-repellent numbers from domain
            domain = [0.8, 1, 1.2]
            gen = RepellentGenerator(domain)
            self._rep_scalers = list(gen.yield_from_domain(self.duration))
        else:
            if len(self.rep_scalers) != self.duration:
                raise ProgramError(
                    'Length of `rep_scalers` must match program duration.')
            self._rep_scalers = self.rep_scalers

        # Set default value for intensity_scalers if None
        if self.intensity_scalers is None:
            # Draw self-repellent numbers from domain
            domain = [0.95, 1, 1.05]
            gen = RepellentGenerator(domain)
            self._intensity_scalers = list(gen.yield_from_domain(self.duration))
        else:
            if len(self.intensity_scalers) != self.duration:
                raise ProgramError(
                    'Length of `intensity_scalers` must match program duration.')
            self._intensity_scalers = self.intensity_scalers