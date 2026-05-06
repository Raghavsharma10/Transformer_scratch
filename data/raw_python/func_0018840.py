def GET_savedmodifiedconditionitemvalues(self) -> None:
        """ToDo: extend functionality and add tests"""
        dict_ = state.modifiedconditionitemvalues.get(self._id)
        if dict_ is None:
            self.GET_conditionitemvalues()
        else:
            for name, value in dict_.items():
                self._outputs[name] = value