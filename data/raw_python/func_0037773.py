def commands(self):
        """
        Fetch individual SQL commands from a SQL commands containing many commands.

        :return: List of commands
        """
        # Retrieve all commands via split function or splitting on ';'
        print('\tRetrieving commands from', self.sql_script)
        print('\tUsing command splitter algorithm {0}'.format(self.split_algo))

        with Timer('\tRetrieved commands in'):
            # Split commands
            # sqlparse packages split function combined with sql_split function
            if self.split_algo is 'sql_parse':
                commands = SplitCommands(self.sql_script).sql_parse

            # Split on every ';' (unreliable)
            elif self.split_algo is 'simple_split':
                commands = SplitCommands(self.sql_script).simple_split()

            # sqlparse package without additional splitting
            elif self.split_algo is 'sql_parse_nosplit':
                commands = SplitCommands(self.sql_script).sql_parse_nosplit

            # Parse every char of the SQL commands and determine breakpoints
            elif self.split_algo is 'sql_split':
                commands = SplitCommands(self.sql_script).sql_split(disable_tqdm=False)
            else:
                commands = SplitCommands(self.sql_script).sql_split(disable_tqdm=False)

            # remove dbo. prefixes from table names
            cleaned_commands = [com.replace("dbo.", '') for com in commands]
        return cleaned_commands