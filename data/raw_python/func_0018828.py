def GET_parameteritemtypes(self) -> None:
        """Get the types of all current exchange items supposed to change
        the values of |Parameter| objects."""
        for item in state.parameteritems:
            self._outputs[item.name] = self._get_itemtype(item)