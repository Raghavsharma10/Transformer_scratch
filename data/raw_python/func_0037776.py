def _execute_commands_from_dir(self, directory):
        """Re-attempt to split and execute the failed commands"""
        # Get file paths and contents
        commands = get_commands_from_dir(directory)

        # Execute failed commands again
        print('\tAttempting to execute {0} failed commands'.format(len(commands)))
        return self.execute(commands, ignored_commands=None, execute_fails=True)