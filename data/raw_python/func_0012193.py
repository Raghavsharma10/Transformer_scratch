def maybe_canonicalize_exe_path(exe_name, iocontext):
    '''There's a tricky interaction between exe paths and `dir`. Exe paths can
    be relative, and so we have to ask: Is an exe path interpreted relative to
    the parent's cwd, or the child's? The answer is that it's platform
    dependent! >.< (Windows uses the parent's cwd, but because of the
    fork-chdir-exec pattern, Unix usually uses the child's.)

    We want to use the parent's cwd consistently, because that saves the caller
    from having to worry about whether `dir` will have side effects, and
    because it's easy for the caller to use path.join if they want to. That
    means that when `dir` is in use, we need to detect exe names that are
    relative paths, and absolutify them. We want to do that as little as
    possible though, both because canonicalization can fail, and because we
    prefer to let the caller control the child's argv[0].

    We never want to absolutify a name like "emacs", because that's probably a
    program in the PATH rather than a local file. So we look for slashes in the
    name to determine what's a filepath and what isn't. Note that anything
    given as a Path will always have a slash by the time we get here, because
    stringify_with_dot_if_path prepends a ./ to them when they're relative.
    This leaves the case where Windows users might pass a local file like
    "foo.bat" as a string, which we can't distinguish from a global program
    name. However, because the Windows has the preferred "relative to parent's
    cwd" behavior already, this case actually works without our help. (The
    thing Windows users have to watch out for instead is local files shadowing
    global program names, which I don't think we can or should prevent.)'''

    has_sep = (os.path.sep in exe_name
               or (os.path.altsep is not None and os.path.altsep in exe_name))

    if has_sep and iocontext.dir is not None and not os.path.isabs(exe_name):
        return os.path.realpath(exe_name)
    else:
        return exe_name