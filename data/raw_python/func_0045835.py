def start_tty(self, conf, interactive):
        """Startup a tty"""
        try:
            api = conf.harpoon.docker_context_maker().api
            container_id = conf.container_id

            stdin = conf.harpoon.tty_stdin
            stdout = conf.harpoon.tty_stdout
            stderr = conf.harpoon.tty_stderr
            if callable(stdin): stdin = stdin()
            if callable(stdout): stdout = stdout()
            if callable(stderr): stderr = stderr()
            dockerpty.start(api, container_id, interactive=interactive, stdout=stdout, stderr=stderr, stdin=stdin)
        except KeyboardInterrupt:
            pass