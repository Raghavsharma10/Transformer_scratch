def abort(self, msg=None):
        '''Aborting a run. Control for loops. Immediate stop/abort.

        The implementation should stop a run ASAP when this event is set. The run is considered incomplete.
        '''
        if not self.abort_run.is_set():
            if msg:
                logging.error('%s%s Aborting run...', msg, ('' if msg[-1] in punctuation else '.'))
            else:
                logging.error('Aborting run...')
        self.abort_run.set()
        self.stop_run.set()