def run(self, data, store, signal, context, **kwargs):
        """ The main run method of the Python task.

        Args:
            data (:class:`.MultiTaskData`): The data object that has been passed from the
                predecessor task.
            store (:class:`.DataStoreDocument`): The persistent data store object that allows the
                task to store data for access across the current workflow run.
            signal (TaskSignal): The signal object for tasks. It wraps the construction
                and sending of signals into easy to use methods.
            context (TaskContext): The context in which the tasks runs.

        Returns:
            Action (Action): An Action object containing the data that should be passed on
                to the next task and optionally a list of successor tasks that
                should be executed.
        """
        params = self.params.eval(data, store, exclude=['command'])

        capture_stdout = self._callback_stdout is not None or params.capture_stdout
        capture_stderr = self._callback_stderr is not None or params.capture_stderr

        stdout_file = TemporaryFile() if params.capture_stdout else None
        stderr_file = TemporaryFile() if params.capture_stderr else None

        stdout = PIPE if capture_stdout else None
        stderr = PIPE if capture_stderr else None

        # change the user or group under which the process should run
        if params.user is not None or params.group is not None:
            pre_exec = self._run_as(params.user, params.group)
        else:
            pre_exec = None

        # call the command
        proc = Popen(self.params.eval_single('command', data, store),
                     cwd=params.cwd, shell=True, env=params.env,
                     preexec_fn=pre_exec, stdout=stdout, stderr=stderr,
                     stdin=PIPE if params.stdin is not None else None)

        # if input is available, send it to the process
        if params.stdin is not None:
            proc.stdin.write(params.stdin.encode(sys.getfilesystemencoding()))

        # send a notification that the process has been started
        try:
            if self._callback_process is not None:
                self._callback_process(proc.pid, data, store, signal, context)
        except (StopTask, AbortWorkflow):
            proc.terminate()
            raise

        # send the output handling to a thread
        if capture_stdout or capture_stderr:
            output_reader = BashTaskOutputReader(proc, stdout_file, stderr_file,
                                                 self._callback_stdout,
                                                 self._callback_stderr,
                                                 params.refresh_time,
                                                 data, store, signal, context)
            output_reader.start()
        else:
            output_reader = None

        # wait for the process to complete and watch for a stop signal
        while proc.poll() is None or\
                (output_reader is not None and output_reader.is_alive()):
            sleep(params.refresh_time)
            if signal.is_stopped:
                proc.terminate()

        if output_reader is not None:
            output_reader.join()
            data = output_reader.data

            # if a stop or abort exception was raised, stop the bash process and re-raise
            if output_reader.exc_obj is not None:
                if proc.poll() is None:
                    proc.terminate()
                raise output_reader.exc_obj

        # send a notification that the process has completed
        if self._callback_end is not None:
            if stdout_file is not None:
                stdout_file.seek(0)
            if stderr_file is not None:
                stderr_file.seek(0)

            self._callback_end(proc.returncode, stdout_file, stderr_file,
                               data, store, signal, context)

        if stdout_file is not None:
            stdout_file.close()

        if stderr_file is not None:
            stderr_file.close()

        return Action(data)