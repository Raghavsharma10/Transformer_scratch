def replace_simple_tags(s, from_tag='italic', to_tag='i', to_open_tag=None):
    """
    Replace tags such as <italic> to <i>
    This does not validate markup
    """
    if to_open_tag:
        s = s.replace('<' + from_tag + '>', to_open_tag)
    elif to_tag:
        s = s.replace('<' + from_tag + '>', '<' + to_tag + '>')
        s = s.replace('<' + from_tag + '/>', '<' + to_tag + '/>')
    else:
        s = s.replace('<' + from_tag + '>', '')
        s = s.replace('<' + from_tag + '/>', '')

    if to_tag:
        s = s.replace('</' + from_tag + '>', '</' + to_tag + '>')
    else:
        s = s.replace('</' + from_tag + '>', '')

    return s