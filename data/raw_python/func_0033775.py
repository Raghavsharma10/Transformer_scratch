def add_var_opt(self,opt,value,short=False):
    """
    Add a variable (macro) option for this node. If the option
    specified does not exist in the CondorJob, it is added so the submit
    file will be correct when written.
    @param opt: option name.
    @param value: value of the option for this node in the DAG.
    """
    macro = self.__bad_macro_chars.sub( r'', opt )
    self.__opts['macro' + macro] = value
    self.__job.add_var_opt(opt,short)