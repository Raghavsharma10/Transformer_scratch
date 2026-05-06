def GET_parameteritemvalues(self) -> None:
        """Get the values of all |ChangeItem| objects handling |Parameter|
        objects."""
        for item in state.parameteritems:
            self._outputs[item.name] = item.value