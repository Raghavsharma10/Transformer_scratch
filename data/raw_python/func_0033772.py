def add_macro(self,name,value):
    """
    Add a variable (macro) for this node.  This can be different for
    each node in the DAG, even if they use the same CondorJob.  Within
    the CondorJob, the value of the macro can be referenced as
    '$(name)' -- for instance, to define a unique output or error file
    for each node.
    @param name: macro name.
    @param value: value of the macro for this node in the DAG
    """
    macro = self.__bad_macro_chars.sub( r'', name )
    self.__opts[macro] = value