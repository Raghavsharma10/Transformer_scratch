def event_estimation(self, event, logliks, logsumexps, renamed=''):
        """Show the values underlying bayesian modality estimations of an event

        Parameters
        ----------


        Returns
        -------


        Raises
        ------
        """
        plotter = _ModelLoglikPlotter()
        plotter.plot(event, logliks, logsumexps, self.modality_to_color,
                     renamed=renamed)
        return plotter