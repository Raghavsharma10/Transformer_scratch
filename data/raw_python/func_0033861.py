def parse_line(self, line):
    """
    For each line we are passed, call the XML parser. Returns the
    line if we are outside one of the ignored tables, otherwise
    returns the empty string.

    @param line: the line of the LIGO_LW XML file to be parsed
    @type line: string

    @return: the line of XML passed in or the null string
    @rtype: string
    """
    self.__p.Parse(line)
    if self.__in_table:
      self.__silent = 1
    if not self.__silent:
      ret = line
    else:
      ret = ""
    if not self.__in_table:
      self.__silent = 0
    return ret