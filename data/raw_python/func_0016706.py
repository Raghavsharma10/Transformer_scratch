def warn_attribs(loc,
                 node,
                 recognised_attribs,
                 reqd_attribs=None):
    '''
    Error checking of XML input: check that the given node has certain
    required attributes, and does not have any unrecognised
    attributes.

    Arguments:
    - `loc`: a string with some information about the location of the
      error in the XML file
    - `node`: the node to check
    - `recognised_attribs`: a set of node attributes which we know how
      to handle
    - `reqd_attribs`: a set of node attributes which we require to be
      present; if this argument is None, it will take the same value
      as `recognised_attribs`
    '''
    if reqd_attribs is None:
        reqd_attribs = recognised_attribs
    found_attribs = set(node.keys())
    if reqd_attribs - found_attribs:
        print(loc, 'missing <{0}> attributes'.format(node.tag),
              reqd_attribs - found_attribs)
    if found_attribs - recognised_attribs:
        print(loc, 'unrecognised <{0}> properties'.format(node.tag),
              found_attribs - recognised_attribs)