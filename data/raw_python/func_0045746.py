def write_to(output, txt):
    """Write some text to some output"""
    if (isinstance(txt, six.binary_type) or six.PY3 and isinstance(output, StringIO)) or isinstance(output, TextIOWrapper):
        output.write(txt)
    else:
        output.write(txt.encode("utf-8", "replace"))