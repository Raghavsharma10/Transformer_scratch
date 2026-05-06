def participating_ecs(self):
        '''A list of the execution contexts this component is participating in.

        '''
        with self._mutex:
            if not self._participating_ecs:
                self._participating_ecs = [ExecutionContext(ec,
                                    self._obj.get_context_handle(ec)) \
                             for ec in self._obj.get_participating_contexts()]
        return self._participating_ecs