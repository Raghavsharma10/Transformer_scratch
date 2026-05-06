def quote_js(text):
    '''Quotes text to be used as JavaScript string in HTML templates. The
    result doesn't contain surrounding quotes.'''
    if isinstance(text, six.binary_type):
        text = text.decode('utf-8') # for Jinja2 Markup
    text = text.replace('\\', '\\\\');
    text = text.replace('\n', '\\n');
    text = text.replace('\r', '');
    for char in '\'"<>&':
        text = text.replace(char, '\\x{:02x}'.format(ord(char)))
    return text