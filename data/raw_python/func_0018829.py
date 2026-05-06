def GET_conditionitemtypes(self) -> None:
        """Get the types of all current exchange items supposed to change
        the values of |StateSequence| or |LogSequence| objects."""
        for item in state.conditionitems:
            self._outputs[item.name] = self._get_itemtype(item)