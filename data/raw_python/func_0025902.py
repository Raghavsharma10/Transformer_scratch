def which_darwin_linkage(force_otool_check=False):
    """ Convenience function.  Returns one of ('x11', 'aqua') in answer to the
    question of whether this is an X11-linked Python/tkinter, or a natively
    built (framework, Aqua) one.  This is only for OSX.
    This relies on the assumption that on OSX, PyObjC is installed
    in the Framework builds of Python.  If it doesn't find PyObjC,
    this inspects the actual tkinter library binary via otool.

    One driving requirement here is to try to make the determination quickly
    and quietly without actually importing/loading any GUI libraries.  We
    even want to avoid importing tkinter if we can.
    """

    # sanity check
    assert sys.platform=='darwin', 'Incorrect usage, not on OSX'

    # If not forced to run otool, then make some quick and dirty
    # simple checks/assumptions, which do not add to startup time and do not
    # attempt to initialize any graphics.
    if not force_otool_check:

        # There will (for now) only ever be an aqua-linked Python/tkinter
        # when using Ureka on darwin, so this is an easy short-circuit check.
        if 'UR_DIR' in os.environ:
            return "aqua"

        # There will *usually* be PyObjC modules on sys.path on the natively-
        # linked Python. This is assumed to be always correct on Python 2.x, as
        # of 2012.  This is kludgy but quick and effective.
        sp = ",".join(sys.path)
        sp = sp.lower().strip(',')
        if '/pyobjc' in sp or 'pyobjc,' in sp or 'pyobjc/' in sp or sp.endswith('pyobjc'):
            return "aqua"

        # Try one more thing - look for the physical PyObjC install dir under site-packages
        # The assumption above using sys.path does not seem to be correct as of the
        # combination of Python2.7.9/PyObjC3.0.4/2015.
        sitepacksloc = os.path.split(os.__file__)[0]+'/site-packages/objc'
        if os.path.exists(sitepacksloc):
            return "aqua"

        # OK, no trace of PyObjC found - need to fall through to the forced otool check.

    # Use otool shell command
    if PY3K:
        import tkinter as TKNTR
    else:
        import Tkinter as TKNTR
    import subprocess
    try:
        tk_dyn_lib = TKNTR._tkinter.__file__
    except AttributeError: # happens on Ureka
        if 'UR_DIR' in os.environ:
            return 'aqua'
        else:
            return 'unknown'
    libs = subprocess.check_output(('/usr/bin/otool', '-L', tk_dyn_lib)).decode('ascii')
    if libs.find('/libX11.') >= 0:
        return "x11"
    else:
        return "aqua"