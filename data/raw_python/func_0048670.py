def to_plain_text(str):
    '''
    Return a plain-text version of a given string

    This is a dumb approach that tags and then removing entity markers
    but this is fine for the content from biocyc where entities are &beta; etc.
    
    Stripping in this way turns these into plaintext 'beta' which is preferable 
    to unicode
    '''
    
    str = strip_tags_re.sub('', str)
    str = strip_entities_re.sub('', str)
    return str