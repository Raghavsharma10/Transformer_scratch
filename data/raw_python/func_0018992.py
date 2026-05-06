def plot_obsseries(self, **kwargs: Any) -> None:
        """Plot the |IOSequence.series| of the |Obs| sequence object.

        See method |Node.plot_allseries| for further information.
        """
        self.__plot_series([self.sequences.obs], kwargs)