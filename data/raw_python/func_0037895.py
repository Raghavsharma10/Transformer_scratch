def _parse_module_list(module_list):
    '''Loop through all the modules and parse them.'''
    for module_meta in module_list:
        name = module_meta['module']

        # Import & parse module
        module = import_module(name)
        output = parse_module(module)

        # Assign to meta.content
        module_meta['content'] = output