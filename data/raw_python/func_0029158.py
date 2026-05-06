def write_template_to_file(conf, template_body):
    """Writes the template to disk
    """
    template_file_name = _get_stack_name(conf) + '-generated-cf-template.json'
    with open(template_file_name, 'w') as opened_file:
        opened_file.write(template_body)
    print('wrote cf-template for %s to disk: %s' % (
        get_env(), template_file_name))
    return template_file_name