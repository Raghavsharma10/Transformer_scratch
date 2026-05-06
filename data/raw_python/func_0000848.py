def join_sentences(string1, string2, glue='.'):
    "concatenate two sentences together with punctuation glue"
    if not string1 or string1 == '':
        return string2
    if not string2 or string2 == '':
        return string1
    # both are strings, continue joining them together with the glue and whitespace
    new_string = string1.rstrip()
    if not new_string.endswith(glue):
        new_string += glue
    new_string += ' ' + string2.lstrip()
    return new_string