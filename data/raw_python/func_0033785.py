def get_cmd_tuple_list(self):
    """
    Return a list of tuples containg the command line arguments
    """

    # pattern to find DAGman macros
    pat = re.compile(r'\$\((.+)\)')
    argpat = re.compile(r'\d+')

    # first parse the options and replace macros with values
    options = self.job().get_opts()
    macros = self.get_opts()

    cmd_list = []

    for k in options:
      val = options[k]
      m = pat.match(val)
      if m:
        key = m.group(1)
        value = macros[key]

        cmd_list.append(("--%s" % k, str(value)))
      else:
        cmd_list.append(("--%s" % k, str(val)))

    # second parse the short options and replace macros with values
    options = self.job().get_short_opts()

    for k in options:
      val = options[k]
      m = pat.match(val)
      if m:
        key = m.group(1)
        value = macros[key]

        cmd_list.append(("-%s" % k, str(value)))
      else:
        cmd_list.append(("-%s" % k, str(val)))

    # lastly parse the arguments and replace macros with values
    args = self.job().get_args()
    macros = self.get_args()

    for a in args:
      m = pat.match(a)
      if m:
        arg_index = int(argpat.findall(a)[0])
        try:
          cmd_list.append(("%s" % macros[arg_index], ""))
        except IndexError:
          cmd_list.append("")
      else:
        cmd_list.append(("%s" % a, ""))

    return cmd_list