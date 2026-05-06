def GET_savedgetitemvalues(self) -> None:
        """Get the previously saved values of all |GetItem| objects."""
        dict_ = state.getitemvalues.get(self._id)
        if dict_ is None:
            self.GET_getitemvalues()
        else:
            for name, value in dict_.items():
                self._outputs[name] = value