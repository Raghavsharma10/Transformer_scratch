def filled_out_template_str(template, **substitutions):
    '''Return str template with applied substitutions.

    Example:
        >>> template = 'Asyl for {{name}} {{surname}}!'
        >>> filled_out_template_str(template, name='Edward', surname='Snowden')
        'Asyl for Edward Snowden!'

        >>> template = '[[[foo]]] was substituted by {{foo}}'
        >>> filled_out_template_str(template, foo='bar')
        '{{foo}} was substituted by bar'

        >>> template = 'names wrapped by {single} {curly} {braces} {{curly}}'
        >>> filled_out_template_str(template, curly='remains unchanged')
        'names wrapped by {single} {curly} {braces} remains unchanged'
    '''
    template = template.replace('{', '{{')
    template = template.replace('}', '}}')
    template = template.replace('{{{{', '{')
    template = template.replace('}}}}', '}')
    template = template.format(**substitutions)
    template = template.replace('{{', '{')
    template = template.replace('}}', '}')
    template = template.replace('[[[', '{{')
    template = template.replace(']]]', '}}')
    return template