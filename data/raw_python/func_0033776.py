def add_file_opt(self,opt,filename,file_is_output_file=False):
    """
    Add a variable (macro) option for this node. If the option
    specified does not exist in the CondorJob, it is added so the submit
    file will be correct when written. The value of the option is also
    added to the list of input files for the DAX.
    @param opt: option name.
    @param value: value of the option for this node in the DAG.
    @param file_is_output_file: A boolean if the file will be an output file
    instead of an input file.  The default is to have it be an input.
    """
    self.add_var_opt(opt,filename)
    if file_is_output_file: self.add_output_file(filename)
    else: self.add_input_file(filename)