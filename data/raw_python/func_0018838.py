def GET_savedparameteritemvalues(self) -> None:
        """Get the previously saved values of those |ChangeItem| objects
        which are handling |Parameter| objects."""
        dict_ = state.parameteritemvalues.get(self._id)
        if dict_ is None:
            self.GET_parameteritemvalues()
        else:
            for name, value in dict_.items():
                self._outputs[name] = value