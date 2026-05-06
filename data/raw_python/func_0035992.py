def _fork_and_supervise_child(cls):
        """Fork a child and then watch the process group until there are
        no processes in it."""
        pid = os.fork()
        if pid == 0:
            # Fork again but orphan the child this time so we'll have
            # the original parent and the second child which is orphaned
            # so we don't have to worry about it becoming a zombie
            cls._orphan_this_process()
            return
        # Since this process is not going to exit, we need to call
        # os.waitpid() so that the first child doesn't become a zombie
        os.waitpid(pid, 0)

        # Generate a list of PIDs to exclude when checking for processes
        # in the group (exclude all ancestors that are in the group)
        pgid = os.getpgrp()
        exclude_pids = set([0, os.getpid()])
        proc = psutil.Process()
        while os.getpgid(proc.pid) == pgid:
            exclude_pids.add(proc.pid)
            proc = psutil.Process(proc.ppid())

        while True:
            try:
                # Look for other processes in this process group
                group_procs = []
                for proc in psutil.process_iter():
                    try:
                        if (os.getpgid(proc.pid) == pgid and
                                proc.pid not in exclude_pids):
                            # We found a process in this process group
                            group_procs.append(proc)
                    except (psutil.NoSuchProcess, OSError):
                        continue

                if group_procs:
                    psutil.wait_procs(group_procs, timeout=1)
                else:
                    # No processes were found in this process group
                    # so we can exit
                    cls._emit_message(
                        'All children are gone. Parent is exiting...\n')
                    sys.exit(0)
            except KeyboardInterrupt:
                # Don't exit immediatedly on Ctrl-C, because we want to
                # wait for the child processes to finish
                cls._emit_message('\n')
                continue