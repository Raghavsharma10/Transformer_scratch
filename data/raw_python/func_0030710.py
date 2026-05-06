def spawn(cls, executable, args, path, env, spawnProcess=None):
        """
        Run an executable with some arguments in the given working directory with
        the given environment variables.

        Returns a Deferred which fires with a two-tuple of (exit status, output
        list) if the process terminates without timing out or being killed by a
        signal.  Otherwise, the Deferred errbacks with either L{error.TimeoutError}
        if any 10 minute period passes with no events or L{ProcessDied} if it is
        killed by a signal.

        On success, the output list is of two-tuples of (file descriptor, bytes).
        """
        d = defer.Deferred()
        proto = cls(d, filepath.FilePath(path))
        if spawnProcess is None:
            spawnProcess = reactor.spawnProcess
        spawnProcess(
            proto,
            executable,
            [executable] + args,
            path=path,
            env=env,
            childFDs={0: 'w', 1: 'r', 2: 'r',
                      cls.BACKCHANNEL_OUT: 'r',
                      cls.BACKCHANNEL_IN: 'w'})
        return d