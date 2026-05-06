def upgrade_setuptools():
    """
    setuptools 12.2 can trigger a really nasty bug
    that eats all memory, so upgrade it to
    18.8, which is known to be good.
    """
    # Note - I tried including the higher version in
    # setup_requires, but was still able to trigger
    # the bug. - stu.axon
    global MIN_SETUPTOOLS
    r = None
    try:
        r = pkg_resources.require(["setuptools"])[0]
    except DistributionNotFound:
        # ok, setuptools will be installed later
        return

    if StrictVersion(r.version) >= StrictVersion(MIN_SETUPTOOLS):
        return
    else:
        print("Upgrading setuptools...")
        subprocess.call("%s -mpip install 'setuptools>=%s'" % (sys.executable, MIN_SETUPTOOLS), shell=True)