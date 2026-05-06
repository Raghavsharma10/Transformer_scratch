def get_template(template_name,fields=None):
    '''get_template will return a template in the template folder,
    with some substitutions (eg, {'{{ graph | safe }}':"fill this in!"}
    '''
    template = None
    if not template_name.endswith('.html'):
        template_name = "%s.html" %(template_name)
    here = "%s/cli/app/templates" %(get_installdir())
    template_path = "%s/%s" %(here,template_name)
    if os.path.exists(template_path):
        template = ''.join(read_file(template_path))
    if fields is not None:
        for tag,sub in fields.items():
            template = template.replace(tag,sub)
    return template