def GET_save_parameteritemvalues(self) -> None:
        """Save the values of those |ChangeItem| objects which are
        handling |Parameter| objects."""
        for item in state.parameteritems:
            state.parameteritemvalues[self._id][item.name] = item.value.copy()