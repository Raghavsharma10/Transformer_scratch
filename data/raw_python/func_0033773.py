def add_io_macro(self,io,filename):
    """
    Add a variable (macro) for storing the input/output files associated
    with this node.
    @param io: macroinput or macrooutput
    @param filename: filename of input/output file
    """
    io = self.__bad_macro_chars.sub( r'', io )
    if io not in self.__opts:
      self.__opts[io] = filename
    else:
      if filename not in self.__opts[io]:
        self.__opts[io] += ',%s' % filename