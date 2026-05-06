def run(self):
        """
        Starts a development server for the zengine application
        """
        from zengine.wf_daemon import run_workers, Worker

        worker_count = int(self.manager.args.workers or 1)
        if not self.manager.args.daemonize:
            print("Starting worker(s)")

        if worker_count > 1 or self.manager.args.autoreload:
            run_workers(worker_count,
                        self.manager.args.paths.split(' '),
                        self.manager.args.daemonize)
        else:
            worker = Worker()
            worker.run()