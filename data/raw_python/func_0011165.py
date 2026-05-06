def pager(text, color=None):
    """Decide what method to use for paging through text."""
    stdout = _default_text_stdout()
    if not isatty(sys.stdin) or not isatty(stdout):
        return _nullpager(stdout, text, color)
    if 'PAGER' in os.environ:
        if WIN:
            return _tempfilepager(text, os.environ['PAGER'], color)
        return _pipepager(text, os.environ['PAGER'], color)
    if os.environ.get('TERM') in ('dumb', 'emacs'):
        return _nullpager(stdout, text, color)
    if WIN or sys.platform.startswith('os2'):
        return _tempfilepager(text, 'more <', color)
    if hasattr(os, 'system') and os.system('(less) 2>/dev/null') == 0:
        return _pipepager(text, 'less', color)

    import tempfile
    fd, filename = tempfile.mkstemp()
    os.close(fd)
    try:
        if hasattr(os, 'system') and os.system('more "%s"' % filename) == 0:
            return _pipepager(text, 'more', color)
        return _nullpager(stdout, text, color)
    finally:
        os.unlink(filename)