def owned_ecs(self):
        '''A list of the execution contexts owned by this component.'''
        with self._mutex:
            if not self._owned_ecs:
                self._owned_ecs = [ExecutionContext(ec,
                    self._obj.get_context_handle(ec)) \
                    for ec in self._obj.get_owned_contexts()]
        return self._owned_ecs