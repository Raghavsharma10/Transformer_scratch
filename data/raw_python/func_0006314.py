def stop(self, msg=None):
        '''Stopping a run. Control for loops. Gentle stop/abort.

        This event should provide a more gentle abort. The run should stop ASAP but the run is still considered complete.
        '''
        if not self.stop_run.is_set():
            if msg:
                logging.info('%s%s Stopping run...', msg, ('' if msg[-1] in punctuation else '.'))
            else:
                logging.info('Stopping run...')
        self.stop_run.set()