def get_stack_info(frames):
    """
    Given a list of frames, returns a list of stack information
    dictionary objects that are JSON-ready.

    We have to be careful here as certain implementations of the
    _Frame class do not contain the nescesary data to lookup all
    of the information we want.
    """
    __traceback_hide__ = True  # NOQA

    results = []
    for frame_info in frames:
        # Old, terrible API
        if isinstance(frame_info, (list, tuple)):
            frame, lineno = frame_info

        else:
            frame = frame_info
            lineno = frame_info.f_lineno

        # Support hidden frames
        f_locals = getattr(frame, 'f_locals', {})
        if _getitem_from_frame(f_locals, '__traceback_hide__'):
            continue

        f_globals = getattr(frame, 'f_globals', {})
        loader = _getitem_from_frame(f_globals, '__loader__')
        module_name = _getitem_from_frame(f_globals, '__name__')

        f_code = getattr(frame, 'f_code', None)
        if f_code:
            abs_path = frame.f_code.co_filename
            function = frame.f_code.co_name
        else:
            abs_path = None
            function = None

        if lineno:
            lineno -= 1

        if lineno is not None and abs_path:
            context = get_lines_from_file(abs_path, lineno, 3, loader, module_name)
        else:
            context = []

        # Try to pull a relative file path
        # This changes /foo/site-packages/baz/bar.py into baz/bar.py
        try:
            base_filename = sys.modules[module_name.split('.', 1)[0]].__file__
            filename = abs_path.split(base_filename.rsplit('/', 2)[0], 1)[-1][1:]
        except:
            filename = abs_path

        if not filename:
            filename = abs_path

        frame_result = {
            'abs_path': abs_path,
            'filename': filename,
            'module': module_name,
            'function': function,
            'lineno': lineno + 1,
            'context': context,
        }
        results.append(frame_result)
    return results