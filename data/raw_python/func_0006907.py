def extract_words(string):
    '''Extract all alphabetic syllabified forms from 'string'.'''
    return re.findall(r'[%s]+[%s\.]*[%s]+' % (A, A, A), string, flags=FLAGS)