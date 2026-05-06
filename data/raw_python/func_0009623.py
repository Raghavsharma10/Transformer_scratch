def _register_array_types(self, objects):
    '''Add array type definitions to internal registry
    
    Args:
      objects (list of VhdlType or VhdlSubtype): Array types to track
    '''
    # Add all array types directly
    types = [o for o in objects if isinstance(o, VhdlType) and o.type_of == 'array_type']
    for t in types:
      self.array_types.add(t.name)

    subtypes = {o.name:o.base_type for o in objects if isinstance(o, VhdlSubtype)}

    # Find all subtypes of an array type
    for k,v in subtypes.iteritems():
      while v in subtypes: # Follow subtypes of subtypes
        v = subtypes[v]
      if v in self.array_types:
        self.array_types.add(k)