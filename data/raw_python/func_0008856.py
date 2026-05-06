def _fit_islands(self, islands):
        """
        Execute fitting on a list of islands
        This function just wraps around fit_island, so that when we do multiprocesing
        a single process will fit multiple islands before returning results.


        Parameters
        ----------
        islands : list of :class:`AegeanTools.models.IslandFittingData`
            The islands to be fit.

        Returns
        -------
        sources : list
            The sources that were fit.
        """
        self.log.debug("Fitting group of {0} islands".format(len(islands)))
        sources = []
        for island in islands:
            res = self._fit_island(island)
            sources.extend(res)
        return sources