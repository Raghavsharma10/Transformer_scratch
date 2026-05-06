def get_lines_from_file(filename, lineno, context_lines, loader=None, module_name=None):
    """
    Returns context_lines before and after lineno from file.
    Returns (pre_context_lineno, pre_context, context_line, post_context).
    """
    source = None
    if loader is not None and hasattr(loader, "get_source"):
        try:
            source = loader.get_source(module_name)
        except ImportError:
            # Traceback (most recent call last):
            #   File "/Users/dcramer/Development/django-sentry/sentry/client/handlers.py", line 31, in emit
            #     get_client().create_from_record(record, request=request)
            #   File "/Users/dcramer/Development/django-sentry/sentry/client/base.py", line 325, in create_from_record
            #     data['__sentry__']['frames'] = varmap(shorten, get_stack_info(stack))
            #   File "/Users/dcramer/Development/django-sentry/sentry/utils/stacks.py", line 112, in get_stack_info
            #     pre_context_lineno, pre_context, context_line, post_context = get_lines_from_file(filename, lineno, 7, loader, module_name)
            #   File "/Users/dcramer/Development/django-sentry/sentry/utils/stacks.py", line 24, in get_lines_from_file
            #     source = loader.get_source(module_name)
            #   File "/System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/pkgutil.py", line 287, in get_source
            #     fullname = self._fix_name(fullname)
            #   File "/System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/pkgutil.py", line 262, in _fix_name
            #     "module %s" % (self.fullname, fullname))
            # ImportError: Loader for module cProfile cannot handle module __main__
            source = None
        if source is not None:
            source = source.splitlines()

    if source is None:
        try:
            f = open(filename)
            try:
                source = f.readlines()
            finally:
                f.close()
        except (OSError, IOError):
            pass

    if source is None:
        return []

    encoding = 'ascii'
    for line in source[:2]:
        # File coding may be specified. Match pattern from PEP-263
        # (http://www.python.org/dev/peps/pep-0263/)
        match = _coding_re.search(line)
        if match:
            encoding = match.group(1)
            break
    source = [unicode(sline, encoding, 'replace') for sline in source]

    lower_bound = max(0, lineno - context_lines)
    upper_bound = min(lineno + context_lines + 1, len(source))

    try:
        return [(lineno + 1, source[lineno].strip('\n')) for lineno in xrange(lower_bound, upper_bound)]
    except IndexError:
        # the file may have changed since it was loaded into memory
        return []