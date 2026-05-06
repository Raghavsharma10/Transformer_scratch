def with_(context, nodelist, val):
    """
    Adds a value to the context (inside of this block) for caching and easy
    access.

    For example::

        {% with person.some_sql_method as total %}
            {{ total }} object{{ total|pluralize }}
        {% endwith %}
    """
    context.push()
    context[self.name] = val
    output = nodelist.render(context)
    context.pop()
    return output