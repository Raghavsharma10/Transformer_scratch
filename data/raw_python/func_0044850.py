def attach_container(self, path=None, save="all",
                         mode="w", nbuffer=50, force=False):
        """add a Container to the simulation which allows some
        persistance to the simulation.

        Parameters
        ----------
        path : str or None (default: None)
            path for the container. If None (the default), the data lives only
            in memory (and are available with `simulation.container`)
        mode : str, optional
            "a" or "w" (default "w")
        save : str, optional
            "all" will save every time-step,
            "last" will only get the last time step
        nbuffer : int, optional
            wait until nbuffer data in the Queue before save on disk.
        timeout : int, optional
            wait until timeout since last flush before save on disk.
        force : bool, optional (default False)
            if True, remove the target folder if not empty. if False, raise an
            error.
        """
        self._container = TriflowContainer("%s/%s" % (path, self.id)
                                           if path else None,
                                           save=save,
                                           mode=mode, metadata=self.parameters,
                                           force=force, nbuffer=nbuffer)
        self._container.connect(self.stream)
        return self._container