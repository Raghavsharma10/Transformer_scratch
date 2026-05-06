def parse_verilog(text):
  '''Parse a text buffer of Verilog code

  Args:
    text (str): Source code to parse
  Returns:
    List of parsed objects.
  '''
  lex = VerilogLexer

  name = None
  kind = None
  saved_type = None
  mode = 'input'
  ptype = 'wire'

  metacomments = []
  parameters = []
  param_items = []

  generics = []
  ports = collections.OrderedDict()
  sections = []
  port_param_index = 0
  last_item = None
  array_range_start_pos = 0

  objects = []

  for pos, action, groups in lex.run(text):
    if action == 'metacomment':
      if last_item is None:
        metacomments.append(groups[0])
      else:
        last_item.desc = groups[0]

    if action == 'section_meta':
      sections.append((port_param_index, groups[0]))

    elif action == 'module':
      kind = 'module'
      name = groups[0]
      generics = []
      ports = collections.OrderedDict()
      param_items = []
      sections = []
      port_param_index = 0

    elif action == 'parameter_start':
      net_type, vec_range = groups

      new_ptype = ''
      if net_type is not None:
        new_ptype += net_type

      if vec_range is not None:
        new_ptype += ' ' + vec_range

      ptype = new_ptype

    elif action == 'param_item':
      generics.append(VerilogParameter(groups[0], 'in', ptype))

    elif action == 'module_port_start':
      new_mode, net_type, signed, vec_range = groups

      new_ptype = ''
      if net_type is not None:
        new_ptype += net_type

      if signed is not None:
        new_ptype += ' ' + signed

      if vec_range is not None:
        new_ptype += ' ' + vec_range

      # Complete pending items
      for i in param_items:
        ports[i] = VerilogParameter(i, mode, ptype)

      param_items = []
      if len(ports) > 0:
        last_item = next(reversed(ports))

      # Start with new mode
      mode = new_mode
      ptype = new_ptype

    elif action == 'port_param':
      ident = groups[0]

      param_items.append(ident)
      port_param_index += 1

    elif action == 'end_module':
      # Finish any pending ports
      for i in param_items:
        ports[i] = VerilogParameter(i, mode, ptype)

      vobj = VerilogModule(name, ports.values(), generics, dict(sections), metacomments)
      objects.append(vobj)
      last_item = None
      metacomments = []

  return objects