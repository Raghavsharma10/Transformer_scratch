def run(self, command, arguments=(), console_mode_stdin=True, skip_cmd_shell=False):
        """This function does something.
        :param command: The command to be executed
        :type name: str.
        :param arguments: A list of arguments to be passed to the command
        :type state: str.
        :returns:  int -- the return code.
        :raises: AttributeError, KeyError

        iclegg: blocking i/o operations are slow, doesnt Python have a moden 'async' mechanism
        rather than replying on 80's style callbacks?
        """
        logging.info('running command: ' + command)
        resource = ResourceLocator(CommandShell.ShellResource)
        resource.add_selector('ShellId', self.__shell_id)
        resource.add_option('WINRS_SKIP_CMD_SHELL', ['FALSE', 'TRUE'][bool(skip_cmd_shell)], True)
        resource.add_option('WINRS_CONSOLEMODE_STDIN', ['FALSE', 'TRUE'][bool(console_mode_stdin)], True)

        command = OrderedDict([('rsp:Command', command)])
        command['rsp:Arguments'] = list(arguments)

        response = self.session.command(resource, {'rsp:CommandLine': command})
        command_id = response['rsp:CommandResponse']['rsp:CommandId']
        logging.info('receive command: ' + command_id)
        return command_id