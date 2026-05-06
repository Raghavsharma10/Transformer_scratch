def to_json(data, compress=False):
    """Converts the output of `generate(...)` to formatted json.

    Floats are rounded to three decimals and positional vectors are printed on
    one line with some whitespace buffer.
    """
    return json.compress(data) if compress else json.dumps(data)