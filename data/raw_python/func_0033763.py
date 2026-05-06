def add_var_opt(self, opt, short=False):
    """
    Add a variable (or macro) option to the condor job. The option is added
    to the submit file and a different argument to the option can be set for
    each node in the DAG.
    @param opt: name of option to add.
    """
    if opt not in self.__var_opts:
      self.__var_opts.append(opt)
      macro = self.__bad_macro_chars.sub( r'', opt )
      if short:
        self.add_short_opt(opt,'$(macro' + macro + ')')
      else:
        self.add_opt(opt,'$(macro' + macro + ')')