def set(parser, token):
    """
        Usage:
        {% set templ_tag var1 var2 ... key %}
        {% set variable key %}
        This tag save result of {% templ_tag var1 var2 ... %} to variable with name key,
        Or will save value of variable to new variable with name key.
    """
    bits = token.contents.split(' ')[1:]
    new_token = Token(TOKEN_BLOCK, ' '.join(bits[:-1]))
    if bits[0] in parser.tags:
        func = parser.tags[bits[0]](parser, new_token)
    else:
        func = Variable(bits[0])
    return SetNode(func, bits[-1])