def GET_getitemvalues(self) -> None:
        """Get the values of all |Variable| objects observed by the
        current |GetItem| objects.

        For |GetItem| objects observing time series,
        |HydPyServer.GET_getitemvalues| returns only the values within
        the current simulation period.
        """
        for item in state.getitems:
            for name, value in item.yield_name2value(state.idx1, state.idx2):
                self._outputs[name] = value