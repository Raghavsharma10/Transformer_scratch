def _orphan_this_process(cls, wait_for_parent=False):
        """Orphan the current process by forking and then waiting for
        the parent to exit."""
        # The current PID will be the PPID of the forked child
        ppid = os.getpid()

        pid = os.fork()
        if pid > 0:
            # Exit parent
            sys.exit(0)

        if wait_for_parent and cls._pid_is_alive(ppid, timeout=1):
            raise DaemonError(
                'Parent did not exit while trying to orphan process')