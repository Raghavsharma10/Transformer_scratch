def _init_check_upodates():
    """Sub function for init
    """
    message, count, packages = check_updates()
    if count > 0:
        print(message)
        for pkg in packages:
            print("{0}".format(pkg))
    else:
        print(message)