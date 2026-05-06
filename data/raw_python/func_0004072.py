def getstate(self):
        """
        Returns RUNNING, -1
                COMPLETE, 0
                or
                EXECUTOR_ERROR, 255
        """
        state = "RUNNING"
        exit_code = -1

        exitcode_file = os.path.join(self.workdir, "exit_code")
        pid_file = os.path.join(self.workdir, "pid")

        if os.path.exists(exitcode_file):
            with open(exitcode_file) as f:
                exit_code = int(f.read())
        elif os.path.exists(pid_file):
            with open(pid_file, "r") as pid:
                pid = int(pid.read())
            try:
                (_pid, exit_status) = os.waitpid(pid, os.WNOHANG)
                if _pid != 0:
                    exit_code = exit_status >> 8
                    with open(exitcode_file, "w") as f:
                        f.write(str(exit_code))
                    os.unlink(pid_file)
            except OSError:
                os.unlink(pid_file)
                exit_code = 255

        if exit_code == 0:
            state = "COMPLETE"
        elif exit_code != -1:
            state = "EXECUTOR_ERROR"

        return state, exit_code