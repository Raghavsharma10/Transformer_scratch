def text_with_newlines(text, line_length=78, newline='\n'):
    '''Return text with a `newline` inserted after each `line_length` char.

    Return `text` unchanged if line_length == 0.
    '''
    if line_length > 0:
        if len(text) <= line_length:
            return text
        else:
            return newline.join([text[idx:idx+line_length]
                                 for idx
                                 in range(0, len(text), line_length)])
    else:
        return text