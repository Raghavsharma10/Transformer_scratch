def sub_template(template,template_tag,substitution):
    '''make a substitution for a template_tag in a template
    '''
    template = template.replace(template_tag,substitution)
    return template