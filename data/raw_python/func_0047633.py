def run_command(self, command, shell=True, env=None, execute='/bin/bash',
                    return_code=None):
        """Run a shell command.

        The options available:

            * ``shell`` to be enabled or disabled, which provides the ability
              to execute arbitrary stings or not. if disabled commands must be
              in the format of a ``list``

            * ``env`` is an environment override and or manipulation setting
              which sets environment variables within the locally executed
              shell.

            * ``execute`` changes the interpreter which is executing the
              command(s).

            * ``return_code`` defines the return code that the command must
              have in order to ensure success. This can be a list of return
              codes if multiple return codes are acceptable.

        :param command: ``str``
        :param shell: ``bol``
        :param env: ``dict``
        :param execute: ``str``
        :param return_code: ``int``
        """
        self.log.info('Command: [ %s ]', command)

        if env is None:
            env = os.environ

        if self.debug is False:
            stdout = open(os.devnull, 'wb')
        else:
            stdout = subprocess.PIPE

        if return_code is None:
            return_code = [0]

        stderr = subprocess.PIPE
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            executable=execute,
            env=env,
            shell=shell
        )

        output, error = process.communicate()

        if process.returncode not in return_code:
            self.log.debug('Command Output: %s, Error Msg: %s', output, error)
            return error, False
        else:
            self.log.debug('Command Output: %s', output)
            return output, True