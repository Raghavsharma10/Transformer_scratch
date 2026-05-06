def attach_process(
            self, command, for_legion=False, broken_counter=None, pidfile=None, control=None, daemonize=None,
            touch_reload=None, signal_stop=None, signal_reload=None, honour_stdin=None,
            uid=None, gid=None, new_pid_ns=None, change_dir=None):
        """Attaches a command/daemon to the master process.

        This will allow the uWSGI master to control/monitor/respawn this process.

        http://uwsgi-docs.readthedocs.io/en/latest/AttachingDaemons.html

        :param str|unicode command: The command line to execute.

        :param bool for_legion: Legion daemons will be executed only on the legion lord node,
            so there will always be a single daemon instance running in each legion.
            Once the lord dies a daemon will be spawned on another node.

        :param int broken_counter: Maximum attempts before considering a daemon "broken".

        :param str|unicode pidfile: The pidfile path to check (enable smart mode).

        :param bool control: If True, the daemon becomes a `control` one:
            if it dies the whole uWSGI instance dies.

        :param bool daemonize: Daemonize the process (enable smart2 mode).

        :param list|str|unicode touch_reload: List of files to check:
            whenever they are 'touched', the daemon is restarted

        :param int signal_stop: The signal number to send to the daemon when uWSGI is stopped.

        :param int signal_reload: The signal number to send to the daemon when uWSGI is reloaded.

        :param bool honour_stdin: The signal number to send to the daemon when uWSGI is reloaded.

        :param str|unicode|int uid: Drop privileges to the specified uid.

            .. note:: Requires master running as root.

        :param str|unicode|int gid: Drop privileges to the specified gid.

            .. note:: Requires master running as root.

        :param bool new_pid_ns: Spawn the process in a new pid namespace.

            .. note:: Requires master running as root.

            .. note:: Linux only.

        :param str|unicode change_dir: Use chdir() to the specified directory
            before running the command.

        """
        rule = KeyValue(
            locals(),
            keys=[
                'command', 'broken_counter', 'pidfile', 'control', 'daemonize', 'touch_reload',
                'signal_stop', 'signal_reload', 'honour_stdin',
                'uid', 'gid', 'new_pid_ns', 'change_dir',
            ],
            aliases={
                'command': 'cmd',
                'broken_counter': 'freq',
                'touch_reload': 'touch',
                'signal_stop': 'stopsignal',
                'signal_reload': 'reloadsignal',
                'honour_stdin': 'stdin',
                'new_pid_ns': 'ns_pid',
                'change_dir': 'chdir',
            },
            bool_keys=['control', 'daemonize', 'honour_stdin'],
            list_keys=['touch_reload'],
        )

        prefix = 'legion-' if for_legion else ''

        self._set(prefix + 'attach-daemon2', rule, multi=True)

        return self._section