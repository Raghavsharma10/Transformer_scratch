def script(self, sql_script, split_algo='sql_split', prep_statements=True, dump_fails=True):
        """Wrapper method providing access to the SQLScript class's methods and properties."""
        return Execute(sql_script, split_algo, prep_statements, dump_fails, self)