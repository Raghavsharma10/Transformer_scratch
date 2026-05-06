def execute(self, commands=None, ignored_commands=('DROP', 'UNLOCK', 'LOCK'), execute_fails=True,
                max_executions=MAX_EXECUTION_ATTEMPTS):
        """
        Sequentially execute a list of SQL commands.

        Check if commands property has already been fetched, if so use the
        fetched_commands rather than getting them again.

        :param commands: List of SQL commands
        :param ignored_commands: Boolean, skip SQL commands that begin with 'DROP'
        :param execute_fails: Boolean, attempt to execute failed commands again
        :param max_executions: Int, max number of attempted executions
        :return: Successful and failed commands
        """
        # Break connection
        self._MySQL.disconnect()
        self._execute_iters += 1
        if self._execute_iters > 0:
            print('\tExecuting commands attempt #{0}'.format(self._execute_iters))

        # Retrieve commands from sql_script if no commands are provided
        commands = self.commands if not commands else commands

        # Remove 'DROP' commands
        if ignored_commands:
            commands = filter_commands(commands, ignored_commands)

        # Reestablish connection
        self._MySQL.reconnect()

        # Execute list of commands
        fail, success = self._execute_commands(commands)

        # Dump failed commands to text files
        print('\t' + str(success), 'successful commands')
        if len(fail) > 1 and self._dump_fails:
            # Dump failed commands
            dump_dir = self.dump_commands(fail)

            # Execute failed commands
            if execute_fails and self._execute_iters < max_executions:
                return self._execute_commands_from_dir(dump_dir)
        return fail, success