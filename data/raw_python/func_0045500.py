def highlight_block(context, nodelist, lexer, **kwargs):
    """
    Code is nodelist ``rendered`` in ``context``
    Returns highlighted code ``div`` tag from ``HtmlFormatter``
    Lexer is guessed by ``lexer`` name
    arguments are passed into the formatter

        Syntax::

            {% highlight_block [lexer name] [formatter options] %}
                ... source code ..
            {% endhighlight_block %}

        Example::

            {% highlight_block python linenos=true %}
                print '{{ request.path }}'
            {% endhighlight_block %}
    """
    if highlighter is None:
        return '<pre>%s</pre>' % str(nodelist.render(context) or '')
    return highlighter(nodelist.render(context) or '', get_lexer_by_name(lexer), HtmlFormatter(**kwargs))