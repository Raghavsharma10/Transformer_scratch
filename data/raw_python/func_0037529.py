def underline(text):
    '''Takes a string, and returns it underscored.'''

    text += "\n"
    for i in range(len(text)-1):
        text += "="
    text += "\n"
    return text