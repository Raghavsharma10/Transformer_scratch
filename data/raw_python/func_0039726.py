def available_method(method_name):
    '''ruturn the method for earliest package in ``pkg_preferences``, if package is available (based on :meth:`pkg_available`)'''
    pkg_prefs_copy = list(pkg_prefs)
    if method_name in method_prefs:
        pkg_prefs_copy = [method_prefs[method_name]] + pkg_prefs_copy
    for pkg in pkg_prefs_copy:
        if pkg in pkgs:
            if method_name in dir(pkgs[pkg]):
                return getattr(pkgs[pkg],method_name)
    nl.notify('Error: Could not find implementation of %s on this computer' % (method_name),level=nl.level.error)