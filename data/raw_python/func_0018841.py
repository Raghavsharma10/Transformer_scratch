def GET_save_getitemvalues(self) -> None:
        """Save the values of all current |GetItem| objects."""
        for item in state.getitems:
            for name, value in item.yield_name2value(state.idx1, state.idx2):
                state.getitemvalues[self._id][name] = value