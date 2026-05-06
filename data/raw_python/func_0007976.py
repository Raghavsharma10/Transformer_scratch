def run(self):
        "Get jobs from the queue and perform them as they arrive."
        while 1:
            # Sleep until there is a job to perform.
            job = self.jobs.get()

            # Yawn. Time to get some work done.
            try:
                job.run()
                self.jobs.task_done()
            except TerminationNotice:
                self.jobs.task_done()
                break