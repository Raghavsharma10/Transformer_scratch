def get_opt( self, opt):
    """
    Returns the value associated with the given command line option.
    Returns None if the option does not exist in the options list.
    @param opt: command line option
    """
    if self.__options.has_key(opt):
      return self.__options[opt]
    return None