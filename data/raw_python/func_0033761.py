def add_ini_opts(self, cp, section):
    """
    Parse command line options from a given section in an ini file and
    pass to the executable.
    @param cp: ConfigParser object pointing to the ini file.
    @param section: section of the ini file to add to the options.
    """
    for opt in cp.options(section):
      arg = string.strip(cp.get(section,opt))
      self.__options[opt] = arg