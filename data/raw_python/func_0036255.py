def build_attr_string(attrs, supported=True):
    '''Build a string that will turn any ANSI shell output the desired
    colour.

    attrs should be a list of keys into the term_attributes table.

    '''
    if not supported:
        return ''
    if type(attrs) == str:
        attrs = [attrs]
    result = '\033['
    for attr in attrs:
        result += term_attributes[attr] + ';'
    return result[:-1] + 'm'