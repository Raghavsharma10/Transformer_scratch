def start(self):
        """
        main loop.
        """

        def main_loop():
            while True:
                threadnames = [thread.name for thread in threading.enumerate()]
                for job_name, concrete_job in self.jobs.items():
                    if job_name not in threadnames:
                        new_thread = Executor(
                            name=job_name,
                            job=concrete_job['method'],
                            logger=self.logger,
                            interval=concrete_job['interval']
                        )
                        new_thread.start()
                        new_thread.join(1)
                    else:
                        thread.join(1)

        if not self.args.debug_mode:

            pid_file = pidlockfile.PIDLockFile(self.args.pid_file)

            self.logger.info(
                'blackbird {0} : starting main process'.format(__version__)
            )

            with DaemonContext(
                files_preserve=[logger.get_handler_fp(self.logger)],
                detach_process=self.args.detach_process,
                uid=self.config['global']['user'],
                gid=self.config['global']['group'],
                stdout=None,
                stderr=None,
                pidfile=pid_file
            ):
                main_loop()

        else:
            self.logger.info(
                'blackbird {0} : started main process in debug mode'
                ''.format(__version__)
            )
            main_loop()