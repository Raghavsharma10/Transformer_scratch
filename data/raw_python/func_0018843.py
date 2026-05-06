def GET_save_timegrid(self) -> None:
        """Save the current simulation period."""
        state.timegrids[self._id] = copy.deepcopy(hydpy.pub.timegrids.sim)