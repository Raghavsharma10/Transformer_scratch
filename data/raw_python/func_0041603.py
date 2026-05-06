def decode_mntp(mntp):
    ''' Mount point strings have a unique encoding for whitespace. :-/
        https://stackoverflow.com/a/13576641/450917
        https://stackoverflow.com/a/6117124/450917
    '''
    import re
    replacements = {
        r'\\040': ' ',
        r'\\011': '\t',
        r'\\012': '\n',
        r'\\134': '\\',
    }
    pattern = re.compile('|'.join(replacements.keys()))
    return pattern.sub(lambda m: replacements[re.escape(m.group(0))], mntp)