def _execute_commands(self, commands, fails=False):
        """Execute commands and get list of failed commands and count of successful commands"""
        # Confirm that prepare_statements flag is on
        if self._prep_statements:
            prepared_commands = [prepare_sql(c) for c in tqdm(commands, total=len(commands),
                                                              desc='Prepping SQL Commands')]
            print('\tCommands prepared', len(prepared_commands))
        else:
            prepared_commands = commands

        desc = 'Executing SQL Commands' if not fails else 'Executing Failed SQL Commands'
        fail, success = [], 0
        for command in tqdm(prepared_commands, total=len(prepared_commands), desc=desc):
            # Attempt to execute command and skip command if error is raised
            try:
                self._MySQL.executemore(command)
                success += 1
            except:
                fail.append(command)
        self._MySQL._commit()
        return fail, success