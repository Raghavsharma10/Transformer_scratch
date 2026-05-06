def debug(context):
    """
    Outputs a whole load of debugging information, including the current
    context and imported modules.

    Sample usage::

        <pre>
            {% debug %}
        </pre>
    """

    from pprint import pformat
    output = [pformat(val) for val in context]
    output.append('\n\n')
    output.append(pformat(sys.modules))
    return ''.join(output)