def bits_to_dict(bits):
    """Convert a Django template tag's kwargs into a dictionary of Python types.

    The only necessary types are number, boolean, list, and string.
    http://pygments.org/docs/formatters/#HtmlFormatter

    from: ["style='monokai'", "cssclass='cssclass',", "boolean='true',", 'num=0,', "list='[]'"]
      to: {'style': 'monokai', 'cssclass': 'cssclass', 'boolean': True, 'num': 0, 'list': [],}
    """
    # Strip any trailing commas
    cleaned_bits = [bit[:-1] if bit.endswith(',') else bit for bit in bits]

    # Create dictionary by splitting on equal signs
    options = dict(bit.split('=') for bit in cleaned_bits)

    # Coerce strings of types to Python types
    for key in options:
        if options[key] == "'true'" or options[key] == "'false'":
            options[key] = options[key].title()
        options[key] = ast.literal_eval(options[key])

    return options