def platform():
    """Return platform for the current shell, e.g. windows or unix"""
    executable = parent()
    basename = os.path.basename(executable)
    basename, _ = os.path.splitext(basename)

    if basename in ("bash", "sh"):
        return "unix"
    if basename in ("cmd", "powershell"):
        return "windows"

    raise SystemError("Unsupported shell: %s" % basename)