def do_function(parser, token):
    """
    Performs a defined function on the passed arguments.
    Normally this returns the output of the function into the template.
    If the second to last argument is ``as``, the result of the function is stored in the context and is named whatever the last argument is.

    Syntax::

        {% [function] [var args...] [name=value kwargs...] [as varname] %}

    Examples::

        {% search '^(\d{3})$' 800 as match %}

        {% map sha1 hello world %}

    """
    name, args, kwargs = get_signature(token, True, True)
    return FunctionNode(parser, name, *args, **kwargs)