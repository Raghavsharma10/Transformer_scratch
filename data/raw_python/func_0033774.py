def add_var_condor_cmd(self, command, value):
    """
    Add a variable (macro) condor command for this node. If the command
    specified does not exist in the CondorJob, it is added so the submit file
    will be correct.
    PLEASE NOTE: AS with other add_var commands, the variable must be set for
    all nodes that use the CondorJob instance.
    @param command: command name
    @param value: Value of the command for this node in the DAG/DAX.
    """
    macro = self.__bad_macro_chars.sub( r'', command )
    self.__macros['macro' + macro] = value
    self.__job.add_var_condor_cmd(command)