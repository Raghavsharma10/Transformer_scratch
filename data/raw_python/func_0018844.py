def GET_savedtimegrid(self) -> None:
        """Get the previously saved simulation period."""
        try:
            self._write_timegrid(state.timegrids[self._id])
        except KeyError:
            self._write_timegrid(hydpy.pub.timegrids.init)