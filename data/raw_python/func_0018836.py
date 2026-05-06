def GET_save_conditionvalues(self) -> None:
        """Save the |StateSequence| and |LogSequence| object values of the
        current |HydPy| instance for the current simulation endpoint."""
        state.conditions[self._id] = state.conditions.get(self._id, {})
        state.conditions[self._id][state.idx2] = state.hp.conditions