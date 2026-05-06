def process(self, state):
        """Process a state and return the next state
Usage:

    out = rule_110.process([True, False, True])
    len(out)  # 5, because a False is added to either side
    out == [True, True, True, True, False]
    out = rule_110.process([False, True, False, True])
    len(out)  # still 5, because leading / trailing False's are removed
    out2 = rule_110.process([1, 0, 1])  # Any data type in the list is okay, as
                                        # long as it's boolean value is correct
    out == out2
"""
        if not isinstance(state, list):
            raise TypeError("state must be list")
        if self.finite_canvas:
            state = _crop_list_to_size(state, self.canvas_size)
        else:
            state = _remove_lead_trail_false(state)
            state.insert(0, self.default_val)
            state.append(self.default_val)
        new_state = []
        for i in range(0, len(state)):
            result = _process_cell(i, state, finite=self.finite_canvas)
            new_state.append(self.rules[result])
        return new_state