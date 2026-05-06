def cmd(parent):
    """Determine subshell command for subprocess.call

    Arguments:
        parent (str): Absolute path to parent shell executable

    """

    shell_name = os.path.basename(parent).rsplit(".", 1)[0]

    dirname = os.path.dirname(__file__)

    # Support for Bash
    if shell_name in ("bash", "sh"):
        shell = os.path.join(dirname, "_shell.sh").replace("\\", "/")
        cmd = [parent.replace("\\", "/"), shell]

    # Support for Cmd
    elif shell_name in ("cmd",):
        shell = os.path.join(dirname, "_shell.bat").replace("\\", "/")
        cmd = [parent, "/K", shell]

    # Support for Powershell
    elif shell_name in ("powershell",):
        raise SystemError("Powershell not yet supported")

    # Unsupported
    else:
        raise SystemError("Unsupported shell: %s" % shell_name)

    return cmd