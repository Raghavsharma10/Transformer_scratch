def run_command_as_worker(self, command, after_post_fork_hook=False):
        """Run the specified command as worker.

        :param str|unicode command:

        :param bool after_post_fork_hook: Whether to run it after `post_fork` hook.

        """
        self._set('worker-exec2' if after_post_fork_hook else 'worker-exec', command, multi=True)

        return self._section