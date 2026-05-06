def _setup_environment(self):
        """Setup the environment for the daemon."""
        # Save the original working directory so that reload can launch
        # the new process with the same arguments as the original
        self._orig_workdir = os.getcwd()

        if self.chrootdir is not None:
            try:
                # Change the root directory
                os.chdir(self.chrootdir)
                os.chroot(self.chrootdir)
            except Exception as ex:
                raise DaemonError('Unable to change root directory '
                                  '({error})'.format(error=str(ex)))

        # Prevent the process from generating a core dump
        self._prevent_core_dump()

        try:
            # Switch directories
            os.chdir(self.workdir)
        except Exception as ex:
            raise DaemonError('Unable to change working directory '
                              '({error})'.format(error=str(ex)))

        # Create the directory for the pid file if necessary
        self._setup_piddir()

        try:
            # Set file creation mask
            os.umask(self.umask)
        except Exception as ex:
            raise DaemonError('Unable to change file creation mask '
                              '({error})'.format(error=str(ex)))

        try:
            # Switch users
            os.setgid(self.gid)
            os.setuid(self.uid)
        except Exception as ex:
            raise DaemonError('Unable to setuid or setgid '
                              '({error})'.format(error=str(ex)))