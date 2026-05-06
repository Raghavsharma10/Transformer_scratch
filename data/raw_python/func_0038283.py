def execute_script(self, sql_script=None, commands=None, split_algo='sql_split', prep_statements=False,
                       dump_fails=True, execute_fails=True, ignored_commands=('DROP', 'UNLOCK', 'LOCK')):
        """Wrapper method for SQLScript class."""
        ss = Execute(sql_script, split_algo, prep_statements, dump_fails, self)
        ss.execute(commands, ignored_commands=ignored_commands, execute_fails=execute_fails)