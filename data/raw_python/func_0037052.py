def os_info():
    """Get OS info
    """
    stype = ""
    slack, ver = slack_ver()
    mir = mirror()
    if mir:
        if "current" in mir:
            stype = "Current"
        else:
            stype = "Stable"
    info = (
        "User: {0}\n"
        "OS: {1}\n"
        "Version: {2}\n"
        "Type: {3}\n"
        "Arch: {4}\n"
        "Kernel: {5}\n"
        "Packages: {6}".format(getpass.getuser(), slack, ver, stype,
                               os.uname()[4], os.uname()[2], ins_packages()))
    return info