def GET_save_modifiedconditionitemvalues(self) -> None:
        """ToDo: extend functionality and add tests"""
        for item in state.conditionitems:
            state.modifiedconditionitemvalues[self._id][item.name] = \
                list(item.device2target.values())[0].value