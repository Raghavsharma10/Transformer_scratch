def do_block(parser, token):
    """
    Process several nodes inside a single block
    Block functions take ``context``, ``nodelist`` as first arguments
    If the second to last argument is ``as``, the rendered result is stored in the context and is named whatever the last argument is.

    Syntax::

        {% [block] [var args...] [name=value kwargs...] [as varname] %}
            ... nodelist ...
        {% end[block] %}

    Examples::

        {% render_block as rendered_output %}
            {{ request.path }}/blog/{{ blog.slug }}
        {% endrender_block %}

        {% highlight_block python %}
            import this
        {% endhighlight_block %}

    """
    name, args, kwargs = get_signature(token, contextable=True)
    kwargs['nodelist'] = parser.parse(('end%s' % name,))
    parser.delete_first_token()
    return BlockNode(parser, name, *args, **kwargs)