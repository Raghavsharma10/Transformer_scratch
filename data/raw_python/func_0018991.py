def plot_simseries(self, **kwargs: Any) -> None:
        """Plot the |IOSequence.series| of the |Sim| sequence object.

        See method |Node.plot_allseries| for further information.
        """
        self.__plot_series([self.sequences.sim], kwargs)