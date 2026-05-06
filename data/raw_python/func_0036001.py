def reload(self):
        """Make the daemon reload itself."""
        pid = self._read_pidfile()
        if pid is None or pid != os.getpid():
            raise DaemonError(
                'Daemon.reload() should only be called by the daemon process '
                'itself')

        # Copy the current environment
        new_environ = os.environ.copy()
        new_environ['DAEMONOCLE_RELOAD'] = 'true'
        # Start a new python process with the same arguments as this one
        subprocess.call(
            [sys.executable] + sys.argv, cwd=self._orig_workdir,
            env=new_environ)
        # Exit this process
        self._shutdown('Shutting down for reload')