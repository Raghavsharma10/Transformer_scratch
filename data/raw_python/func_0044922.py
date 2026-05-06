def filter(context, nodelist, filter_exp):
    """
    Filters the contents of the block through variable filters.

    Filters can also be piped through each other, and they can have
    arguments -- just like in variable syntax.

    Sample usage::

        {% filter force_escape|lower %}
            This text will be HTML-escaped, and will appear in lowercase.
        {% endfilter %}
    """
    output = nodelist.render(context)
    # Apply filters.
    context.update({'var': output})
    filtered = filter_expr.resolve(context)
    context.pop()
    return filtered