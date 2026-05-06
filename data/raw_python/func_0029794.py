def color_print(s, color=None, highlight=None, end='\n', file=sys.stdout,
                **kwargs):
    """
    From http://stackoverflow.com/a/287944/610569
    See also https://gist.github.com/Sheljohn/68ca3be74139f66dbc6127784f638920
    """
    if color in palette and color != 'default':
        s = palette[color] + s
    # Highlight / Background color.
    if highlight and highlight in highlighter:
        s = highlighter[highlight] + s
    # Custom string format.
    for name, value in kwargs.items():
        if name in formatter and value == True:
            s = formatter[name] + s
    print(s + palette['default'], end=end, file=file)