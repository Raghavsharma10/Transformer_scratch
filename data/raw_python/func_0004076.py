def getstate(self):
        """
        Returns QUEUED,          -1
                INITIALIZING,    -1
                RUNNING,         -1
                COMPLETE,         0
                or
                EXECUTOR_ERROR, 255
        """
        # the jobstore never existed
        if not os.path.exists(self.jobstorefile):
            logging.info('Workflow ' + self.run_id + ': QUEUED')
            return "QUEUED", -1

        # completed earlier
        if os.path.exists(self.statcompletefile):
            logging.info('Workflow ' + self.run_id + ': COMPLETE')
            return "COMPLETE", 0

        # errored earlier
        if os.path.exists(self.staterrorfile):
            logging.info('Workflow ' + self.run_id + ': EXECUTOR_ERROR')
            return "EXECUTOR_ERROR", 255

        # the workflow is staged but has not run yet
        if not os.path.exists(self.errfile):
            logging.info('Workflow ' + self.run_id + ': INITIALIZING')
            return "INITIALIZING", -1

        # TODO: Query with "toil status"
        completed = False
        with open(self.errfile, 'r') as f:
            for line in f:
                if 'Traceback (most recent call last)' in line:
                    logging.info('Workflow ' + self.run_id + ': EXECUTOR_ERROR')
                    open(self.staterrorfile, 'a').close()
                    return "EXECUTOR_ERROR", 255
                # run can complete successfully but fail to upload outputs to cloud buckets
                # so save the completed status and make sure there was no error elsewhere
                if 'Finished toil run successfully.' in line:
                    completed = True
        if completed:
            logging.info('Workflow ' + self.run_id + ': COMPLETE')
            open(self.statcompletefile, 'a').close()
            return "COMPLETE", 0

        logging.info('Workflow ' + self.run_id + ': RUNNING')
        return "RUNNING", -1