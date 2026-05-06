def run_primlist(self, primlist, skip_remaining=False):
        '''Runs runs from a primlist.

        Parameters
        ----------
        primlist : string
            Filename of primlist.
        skip_remaining : bool
            If True, skip remaining runs, if a run does not exit with status FINISHED.

        Note
        ----
        Primlist is a text file of the following format (comment line by adding '#'):
        <module name (containing class) or class (in either case use dot notation)>; <scan parameter>=<value>; <another scan parameter>=<another value>
        '''
        runlist = self.open_primlist(primlist)
        for index, run in enumerate(runlist):
            logging.info('Progressing with run %i out of %i...', index + 1, len(runlist))
            join = self.run_run(run, use_thread=True)
            status = join()
            if skip_remaining and not status == run_status.finished:
                logging.error('Exited run %i with status %s: Skipping all remaining runs.', run.run_number, status)
                break