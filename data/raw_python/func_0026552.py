def _construct_module(info, target):
    """Build a module from templates and user supplied information"""

    for path in paths:
        real_path = os.path.abspath(os.path.join(target, path.format(**info)))
        log("Making directory '%s'" % real_path)
        os.makedirs(real_path)

    # pprint(info)
    for item in templates.values():
        source = os.path.join('dev/templates', item[0])
        filename = os.path.abspath(
            os.path.join(target, item[1].format(**info)))
        log("Creating file from template '%s'" % filename,
            emitter='MANAGE')
        write_template_file(source, filename, info)