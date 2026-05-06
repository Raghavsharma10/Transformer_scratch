def exec_command(self):
        """Glean the command to run and exec.

        On problems, sys.exit.
        This method should *never* return.
        """
        if not self.original_command_string:
            raise SSHEnvironmentError('no SSH command found; '
                                      'interactive shell disallowed.')

        command_info = {'from': self.get_client_ip(),
                        'keyname': self.keyname,
                        'ssh_original_comand': self.original_command_string,
                        'time': time.time()}

        os.environ['AUTHPROGS_KEYNAME'] = self.keyname

        retcode = 126
        try:
            match = self.find_match()
            command_info['command'] = match.get('command')
            self.logdebug('find_match returned "%s"\n' % match)

            command = match['command']
            retcode = subprocess.call(command)
            command_info['code'] = retcode
            self.log('result: %s\n' % command_info)
            sys.exit(retcode)
        except (CommandRejected, OSError) as err:
            command_info['exception'] = '%s' % err
            self.log('result: %s\n' % command_info)
            sys.exit(retcode)