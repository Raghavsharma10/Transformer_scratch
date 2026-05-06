def generic_insert_module(module_name, args, **kwargs):
    """
    In general we have a initial template and then insert new data, so we dont repeat the schema for each module
    :param module_name: String with module name
    :paran **kwargs: Args to be rendered in template
    """
    file = create_or_open(
        '{}.py'.format(module_name), 
        os.path.join(
            BASE_TEMPLATES_DIR, 
            '{}_initial.py.tmpl'.format(module_name)
        ), 
        args
    )
        
    render_template_with_args_in_file(
        file, 
        os.path.join(
            BASE_TEMPLATES_DIR, 
            '{}.py.tmpl'.format(module_name)
        ), 
        **kwargs
        )
    file.close()