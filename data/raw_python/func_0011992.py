def package_config(path, template='__config__.ini.TEMPLATE', config_name='__config__.ini', **params):
    """configure the module at the given path with a config template and file.
        path        = the filesystem path to the given module
        template    = the config template filename within that path
        config_name = the config filename within that path
        params      = a dict containing config params, which are found in the template using %(key)s.
    """
    config_fns = []
    template_fns = rglob(path, template)
    for template_fn in template_fns:
        config_template = ConfigTemplate(fn=template_fn)
        config = config_template.render(
            fn=os.path.join(os.path.dirname(template_fn), config_name), 
            prompt=True, path=path, **params)
        config.write()
        config_fns.append(config.fn)
        log.info('wrote %r' % config)
    return config_fns