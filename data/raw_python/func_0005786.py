def attach_process_classic(self, command_or_pid_path, background, control=False, for_legion=False):
        """Attaches a command/daemon to the master process optionally managed by a pidfile.

        This will allow the uWSGI master to control/monitor/respawn this process.

        .. note:: This uses old classic uWSGI means of process attaching
            To have more control use ``.attach_process()`` method (requires  uWSGI 2.0+)

        http://uwsgi-docs.readthedocs.io/en/latest/AttachingDaemons.html

        :param str|unicode command_or_pid_path:

        :param bool background: Must indicate whether process is in background.

        :param bool control: Consider this process a control: when the daemon dies, the master exits.

            .. note:: pidfile managed processed not supported.

        :param bool for_legion: Legion daemons will be executed only on the legion lord node,
            so there will always be a single daemon instance running in each legion.
            Once the lord dies a daemon will be spawned on another node.

            .. note:: uWSGI 1.9.9+ required.

        """
        prefix = 'legion-' if for_legion else ''

        if '.pid' in command_or_pid_path:

            if background:
                # Attach a command/daemon to the master process managed by a pidfile (the command must daemonize)
                self._set(prefix + 'smart-attach-daemon', command_or_pid_path, multi=True)

            else:
                # Attach a command/daemon to the master process managed by a pidfile (the command must NOT daemonize)
                self._set(prefix + 'smart-attach-daemon2', command_or_pid_path, multi=True)

        else:
            if background:
                raise ConfigurationError('Background flag is only supported for pid-governed commands')

            if control:
                # todo needs check
                self._set('attach-control-daemon', command_or_pid_path, multi=True)

            else:
                # Attach a command/daemon to the master process (the command has to remain in foreground)
                self._set(prefix + 'attach-daemon', command_or_pid_path, multi=True)

        return self._section