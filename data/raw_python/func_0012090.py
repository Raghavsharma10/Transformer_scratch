def filled_out_template(filename, **substitutions):
    '''Return content of file filename with applied substitutions.'''
    res = None
    with open(filename, 'r') as fp:
        template = fp.read()
        res = filled_out_template_str(template, **substitutions)
    return res