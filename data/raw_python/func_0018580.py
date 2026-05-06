def get_msdn_ref(name):
    """ Try and create a reference to a type on MSDN """
    in_msdn = False
    if name in MSDN_VALUE_TYPES:
        name = MSDN_VALUE_TYPES[name]
        in_msdn = True
    if name.startswith('System.'):
        in_msdn = True
    if in_msdn:
        link = name.split('<')[0]
        if link in MSDN_LINK_MAP:
            link = MSDN_LINK_MAP[link]
        else:
            link = link.lower()
        url = 'https://msdn.microsoft.com/en-us/library/'+link+'.aspx'
        node = nodes.reference(name, shorten_type(name))
        node['refuri'] = url
        node['reftitle'] = name
        return node
    else:
        return None