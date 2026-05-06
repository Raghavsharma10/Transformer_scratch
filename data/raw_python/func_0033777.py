def add_var_arg(self, arg):
    """
    Add a variable (or macro) argument to the condor job. The argument is
    added to the submit file and a different value of the argument can be set
    for each node in the DAG.
    @param arg: name of option to add.
    """
    self.__args.append(arg)
    self.__job.add_var_arg(self.__arg_index)
    self.__arg_index += 1