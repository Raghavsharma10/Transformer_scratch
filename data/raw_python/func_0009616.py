def parse_vhdl(text):
  '''Parse a text buffer of VHDL code

  Args:
    text(str): Source code to parse
  Returns:
    Parsed objects.
  '''
  lex = VhdlLexer
  
  name = None
  kind = None
  saved_type = None
  end_param_group = False
  cur_package = None

  metacomments = []
  parameters = []
  param_items = []

  generics = []
  ports = []
  sections = []
  port_param_index = 0
  last_item = None
  array_range_start_pos = 0

  objects = []
  
  for pos, action, groups in lex.run(text):
    if action == 'metacomment':
      realigned = re.sub(r'^#+', lambda m: ' ' * len(m.group(0)), groups[0])
      if last_item is None:
        metacomments.append(realigned)
      else:
        last_item.desc = realigned
    if action == 'section_meta':
      sections.append((port_param_index, groups[0]))

    elif action == 'function':
      kind = 'function'
      name = groups[0]
      param_items = []
      parameters = []
    elif action == 'procedure':
      kind = 'procedure'
      name = groups[0]
      param_items = []
      parameters = []
    elif action == 'param':
      if end_param_group:
        # Complete previous parameters
        for i in param_items:
          parameters.append(i)
        param_items = []
        end_param_group = False

      param_items.append(VhdlParameter(groups[1]))
    elif action == 'param_type':
      mode, ptype = groups
      
      if mode is not None:
        mode = mode.strip()
      
      for i in param_items: # Set mode and type for all pending parameters
        i.mode = mode
        i.data_type = ptype

      end_param_group = True

    elif action == 'param_default':
      for i in param_items:
        i.default_value = groups[0]

    elif action == 'end_subprogram':
      # Complete last parameters
      for i in param_items:
        parameters.append(i)
        
      if kind == 'function':
        vobj = VhdlFunction(name, cur_package, parameters, groups[0], metacomments)
      else:
        vobj = VhdlProcedure(name, cur_package, parameters, metacomments)
      
      objects.append(vobj)
    
      metacomments = []
      parameters = []
      param_items = []
      kind = None
      name = None

    elif action == 'component':
      kind = 'component'
      name = groups[0]
      generics = []
      ports = []
      param_items = []
      sections = []
      port_param_index = 0

    elif action == 'generic_param':
      param_items.append(groups[0])

    elif action == 'generic_param_type':
      ptype = groups[0]
      
      for i in param_items:
        generics.append(VhdlParameter(i, 'in', ptype))
      param_items = []
      last_item = generics[-1]

    elif action == 'port_param':
      param_items.append(groups[0])
      port_param_index += 1

    elif action == 'port_param_type':
      mode, ptype = groups

      for i in param_items:
        ports.append(VhdlParameter(i, mode, ptype))

      param_items = []
      last_item = ports[-1]

    elif action == 'port_array_param_type':
      mode, ptype = groups
      array_range_start_pos = pos[1]

    elif action == 'array_range_end':
      arange = text[array_range_start_pos:pos[0]+1]

      for i in param_items:
        ports.append(VhdlParameter(i, mode, ptype + arange))

      param_items = []
      last_item = ports[-1]

    elif action == 'end_component':
      vobj = VhdlComponent(name, cur_package, ports, generics, dict(sections), metacomments)
      objects.append(vobj)
      last_item = None
      metacomments = []

    elif action == 'package':
      objects.append(VhdlPackage(groups[0]))
      cur_package = groups[0]
      kind = None
      name = None

    elif action == 'type':
      saved_type = groups[0]

    elif action in ('array_type', 'file_type', 'access_type', 'record_type', 'range_type', 'enum_type', 'incomplete_type'):
      vobj = VhdlType(saved_type, cur_package, action, metacomments)
      objects.append(vobj)
      kind = None
      name = None
      metacomments = []

    elif action == 'subtype':
      vobj = VhdlSubtype(groups[0], cur_package, groups[1], metacomments)
      objects.append(vobj)
      kind = None
      name = None
      metacomments = []

    elif action == 'constant':
      vobj = VhdlConstant(groups[0], cur_package, groups[1], metacomments)
      objects.append(vobj)
      kind = None
      name = None
      metacomments = []

  return objects