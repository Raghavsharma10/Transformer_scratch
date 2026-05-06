def highlight(code, lexer, **kwargs):
    """
    Returns highlighted code ``div`` tag from ``HtmlFormatter``
    Lexer is guessed by ``lexer`` name
    arguments are passed into the formatter

        Syntax::

            {% highlight [source code] [lexer name] [formatter options] %}

        Example::

            {% highlight 'print "Hello World"' python linenos=true %}
    """
    if highlighter is None:
        return '<pre>%s</pre>' % code
    return highlighter(code or '', get_lexer_by_name(lexer), HtmlFormatter(**kwargs))