def find_importer_frame():
    """Returns the outer frame importing this "end" module.

    If this module is being imported by other means than import statement,
    None is returned.

    Returns:
        A frame object or None.
    """
    byte = lambda ch: ord(ch) if PY2 else ch
    frame = inspect.currentframe()
    try:
        while frame:
            code = frame.f_code
            lasti = frame.f_lasti
            if byte(code.co_code[lasti]) == dis.opmap['IMPORT_NAME']:
                # FIXME: Support EXTENDED_ARG.
                arg = (
                    byte(code.co_code[lasti + 1])
                    + byte(code.co_code[lasti + 2]) * 256)
                name = code.co_names[arg]
                if name == 'end':
                    break
                end
            end
            frame = frame.f_back
        end
        return frame
    finally:
        del frame
    end