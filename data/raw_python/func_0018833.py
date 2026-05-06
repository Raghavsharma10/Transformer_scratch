def GET_conditionitemvalues(self) -> None:
        """Get the values of all |ChangeItem| objects handling |StateSequence|
        or |LogSequence| objects."""
        for item in state.conditionitems:
            self._outputs[item.name] = item.value