def _reset_file_descriptors(self):
        """Close open file descriptors and redirect standard streams."""
        if self.close_open_files:
            # Attempt to determine the max number of open files
            max_fds = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
            if max_fds == resource.RLIM_INFINITY:
                # If the limit is infinity, use a more reasonable limit
                max_fds = 2048
        else:
            # If we're not closing all open files, we at least need to
            # reset STDIN, STDOUT, and STDERR.
            max_fds = 3

        for fd in range(max_fds):
            try:
                os.close(fd)
            except OSError:
                # The file descriptor probably wasn't open
                pass

        # Redirect STDIN, STDOUT, and STDERR to /dev/null
        devnull_fd = os.open(os.devnull, os.O_RDWR)
        os.dup2(devnull_fd, 0)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)