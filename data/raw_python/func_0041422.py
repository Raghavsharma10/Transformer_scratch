def universal_read(fname):
    '''Will open and read a file with universal line endings, trying to decode whatever format it's in (e.g., utf8 or utf16)'''
    with open(fname,'rU') as f:
        data = f.read()
    enc_guess = chardet.detect(data)
    return data.decode(enc_guess['encoding'])