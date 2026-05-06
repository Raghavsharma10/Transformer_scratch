def add_var_condor_cmd(self, command):
    """
    Add a condor command to the submit file that allows variable (macro)
    arguments to be passes to the executable.
    """
    if command not in self.__var_cmds:
        self.__var_cmds.append(command)
        macro = self.__bad_macro_chars.sub( r'', command )
        self.add_condor_cmd(command, '$(macro' + macro + ')')