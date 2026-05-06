def _run(self):
        '''The actor's main work loop'''
      
        while self._is_running:
            yield from self._task()

        # Signal that the loop has finished.
        self._run_complete.set_result(True)