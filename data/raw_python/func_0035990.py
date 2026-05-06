def _detach_process(self):
        """Detach the process via the standard double-fork method with
        some extra magic."""
        # First fork to return control to the shell
        pid = os.fork()
        if pid > 0:
            # Wait for the first child, because it's going to wait and
            # check to make sure the second child is actually running
            # before exiting
            os.waitpid(pid, 0)
            sys.exit(0)

        # Become a process group and session group leader
        os.setsid()

        # Fork again so the session group leader can exit and to ensure
        # we can never regain a controlling terminal
        pid = os.fork()
        if pid > 0:
            time.sleep(1)
            # After waiting one second, check to make sure the second
            # child hasn't become a zombie already
            status = os.waitpid(pid, os.WNOHANG)
            if status[0] == pid:
                # The child is already gone for some reason
                exitcode = status[1] % 255
                self._emit_failed()
                self._emit_error('Child exited immediately with exit '
                                 'code {code}'.format(code=exitcode))
                sys.exit(exitcode)
            else:
                self._emit_ok()
                sys.exit(0)

        self._reset_file_descriptors()