def do_comparison(parser, token):
    """
    Compares passed arguments. 
    Attached functions should return boolean ``True`` or ``False``.
    If the attached function returns ``True``, the first node list is rendered.
    If the attached function returns ``False``, the second optional node list is rendered (part after the ``{% else %}`` statement). 
    If the last argument in the tag is ``negate``, then the opposite node list is rendered (like an ``ifnot`` tag).
    
    Syntax::

        {% if_[comparison] [var args...] [name=value kwargs...] [negate] %}
            {# first node list in here #}
        {% else %}
            {# second optional node list in here #}
        {% endif_[comparison] %}


    Supported comparisons are ``match``, ``find``, ``startswith``, ``endswith``,
    ``less``, ``less_or_equal``, ``greater`` and ``greater_or_equal`` and many more.
    Checkout the :ref:`contrib-index` for more examples

    Examples::

        {% if_less some_object.id 3 %}
        {{ some_object }} has an id less than 3.
        {% endif_less %}

        {% if_match request.path '^/$' %}
        Welcome home
        {% endif_match %}

    """
    name, args, kwargs = get_signature(token, comparison=True)
    name = name.replace('if_if', 'if')
    end_tag = 'end' + name
    kwargs['nodelist_true'] = parser.parse(('else', end_tag))
    token = parser.next_token()
    if token.contents == 'else':
        kwargs['nodelist_false'] = parser.parse((end_tag,))
        parser.delete_first_token()
    else:
        kwargs['nodelist_false'] = template.NodeList()
    if name.startswith('if_'):
        name = name.split('if_')[1]
    return ComparisonNode(parser, name, *args, **kwargs)