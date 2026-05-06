def _append_condition(self, sinon_stub_condition, func):
        '''
        Permanently saves the current (volatile) conditions, which would be otherwise lost

        In the _conditions dictionary, the keys "args", "kwargs", "oncall" and "action"
        each refer to a list. All 4 lists have a value appended each time the user calls
        returns or throws to add a condition to the stub. Hence, all 4 lists are in sync,
        so a single index refers to the same condition in all 4 lists.

        e.g.
            stub.withArgs(5).returns(7)
              # conditions: args [(5,)] kwargs [()] oncall [None] action [7]
            stub.withArgs(10).onFirstCall().returns(14)
              # conditions: args [(5,),(10,)] kwargs [(),()] oncall [None,1] action [7,14]

        Args:
            sinon_stub_condition: the _SinonStubCondition object that holds the current conditions
            func: returns a value or raises an exception (i.e. the action to take, as specified by the user)
        '''
        self._conditions["args"].append(sinon_stub_condition._cond_args)
        self._conditions["kwargs"].append(sinon_stub_condition._cond_kwargs)
        self._conditions["oncall"].append(sinon_stub_condition._oncall)
        self._conditions["action"].append(func)