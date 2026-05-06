def _format_newlines(prefix, formatted_node, options):
    """
    Convert newlines into U+23EC characters, followed by an actual newline and
    then a tree prefix so as to position the remaining text under the previous
    line.
    """
    replacement = u''.join([
        options.NEWLINE,
        u'\n',
        prefix])
    return formatted_node.replace(u'\n', replacement)