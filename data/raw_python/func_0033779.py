def write_vars(self,fh):
    """
    Write the variable (macro) options and arguments to the DAG file
    descriptor.
    @param fh: descriptor of open DAG file.
    """
    if self.__macros.keys() or self.__opts.keys() or self.__args:
      fh.write( 'VARS ' + self.__name )
    for k in self.__macros.keys():
      fh.write( ' ' + str(k) + '="' + str(self.__macros[k]) + '"' )
    for k in self.__opts.keys():
      fh.write( ' ' + str(k) + '="' + str(self.__opts[k]) + '"' )
    if self.__args:
      for i in range(self.__arg_index):
        fh.write( ' macroargument' + str(i) + '="' + self.__args[i] + '"' )
    fh.write( '\n' )