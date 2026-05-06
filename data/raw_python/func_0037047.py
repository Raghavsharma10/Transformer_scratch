def ins_packages():
    """Count installed Slackware packages
    """
    count = 0
    for pkg in os.listdir(pkg_path):
        if not pkg.startswith("."):
            count += 1
    return count