def hexify(message):
    '''Print out printable characters, but others in hex'''
    import string
    hexified = []
    for char in message:
        if (char in '\n\r \t') or (char not in string.printable):
            hexified.append('\\x%02x' % ord(char))
        else:
            hexified.append(char)
    return ''.join(hexified)