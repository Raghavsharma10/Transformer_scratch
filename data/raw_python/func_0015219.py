def run_command(cls,
                    cmd_str,
                    log_level=logging.DEBUG,
                    ignore_sigint=False,
                    output_callback=None,
                    as_user=None,
                    log_secret=False,
                    env=None):
        """Runs a command from string, e.g. "cp foo bar"
        Args:
            cmd_str: the command to run as string
            log_level: level at which to log command output (DEBUG by default)
            ignore_sigint: should we ignore sigint during this command (False by default)
            output_callback: function that gets called with every line of output as argument
            as_user: run as specified user (the best way to do this will be deduced by DA)
                runs as current user if as_user == None
            log_secret: if True, the command invocation will only be logged as
                "LOGGING PREVENTED FOR SECURITY REASONS", no output will be logged
            env: if not None, pass to subprocess as shell environment; else use
                original DevAssistant environment
        """
        # run format processors on cmd_str
        for name, cmd_proc in cls.command_processors.items():
            cmd_str = cmd_proc(cmd_str)

        # TODO: how to do cd with as_user?
        if as_user and not cmd_str.startswith('cd '):
            cmd_str = cls.format_for_another_user(cmd_str, as_user)
        cls.log(log_level, cmd_str, 'cmd_call', log_secret)

        if cmd_str.startswith('cd '):
            # special-case cd to behave like shell cd and stay in the directory
            try:
                directory = cmd_str[3:]
                # delete any quotes, os.chdir doesn't split words like sh does
                if directory[0] == directory[-1] == '"':
                    directory = directory[1:-1]
                os.chdir(directory)
            except OSError as e:
                raise exceptions.ClException(cmd_str, 1, six.text_type(e))
            return ''

        stdin_pipe = None
        stdout_pipe = subprocess.PIPE
        stderr_pipe = subprocess.STDOUT
        preexec_fn = cls.ignore_sigint if ignore_sigint else None
        env = os.environ if env is None else env
        proc = subprocess.Popen(cmd_str,
                                stdin=stdin_pipe,
                                stdout=stdout_pipe,
                                stderr=stderr_pipe,
                                shell=True,
                                preexec_fn=preexec_fn,
                                env=env)
        # register process to cls.subprocesses
        cls.subprocesses[proc.pid] = proc

        stdout = []
        while proc.poll() is None:
            try:
                output = proc.stdout.readline().decode(utils.defenc)
                if output:
                    output = output.strip()
                    stdout.append(output)
                    cls.log(log_level, output, 'cmd_out', log_secret)
                if output_callback:
                    output_callback(output)
            except IOError as e:
                if e.errno == errno.EINTR:  # Interrupted system call in Python 2.6
                    sys.stderr.write('Can\'t interrupt this process!\n')
                else:
                    raise e

        # remove process from cls.subprocesses
        cls.subprocesses.pop(proc.pid)

        # add a newline to the end - if there is more output in output_rest, we'll be appending
        # it line by line; if there's no more output, we strip anyway
        stdout = '\n'.join(stdout) + '\n'
        # there may be some remains not read after exiting the previous loop
        output_rest = proc.stdout.read().strip().decode(utils.defenc)
        # we want to log lines separately, not as one big chunk
        output_rest_lines = output_rest.splitlines()
        for i, l in enumerate(output_rest_lines):
            cls.log(log_level, l, 'cmd_out', log_secret)
            # add newline for every line - for last line, only add it if it was originally present
            if i != len(output_rest_lines) - 1 or output_rest.endswith('\n'):
                l += '\n'
            stdout += l
            if output_callback:
                output_callback(l)

        # log return code always on debug level
        cls.log(logging.DEBUG, proc.returncode, 'cmd_retcode', log_secret)
        stdout = stdout.strip()

        if proc.returncode == 0:
            return stdout
        else:
            raise exceptions.ClException(cmd_str,
                                         proc.returncode,
                                         stdout)