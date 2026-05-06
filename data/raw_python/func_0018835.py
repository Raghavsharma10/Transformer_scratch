def GET_load_conditionvalues(self) -> None:
        """Assign the |StateSequence| or |LogSequence| object values available
        for the current simulation start point to the current |HydPy| instance.

        When the simulation start point is identical with the initialisation
        time point and you did not save conditions for it beforehand, the
        "original" initial conditions are used (normally those of the
        conditions files of the respective *HydPy*  project).
        """
        try:
            state.hp.conditions = state.conditions[self._id][state.idx1]
        except KeyError:
            if state.idx1:
                self._statuscode = 500
                raise RuntimeError(
                    f'Conditions for ID `{self._id}` and time point '
                    f'`{hydpy.pub.timegrids.sim.firstdate}` are required, '
                    f'but have not been calculated so far.')
            else:
                state.hp.conditions = state.init_conditions