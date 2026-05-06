def load_module(ldr, fqname):
    '''Load `fqname` from under `ldr.fspath`.

       The `fqname` argument is the fully qualified module name,
       eg. "spam.eggs.ham".  As explained above, when ::

         finder.find_module("spam.eggs.ham")

       is called, "spam.eggs" has already been imported and added
       to `sys.modules`.  However, the `find_module()` method isn't
       necessarily always called during an actual import:
       meta tools that analyze import dependencies (such as freeze,
       Installer or py2exe) don't actually load modules, so
       a finder shouldn't depend on the parent package being
       available in `sys.modules`.

       The `load_module()` method has a few responsibilities that
       it must fulfill before it runs any code:

       * If there is an existing module object named 'fullname' in
         `sys.modules`, the loader must use that existing module.
         (Otherwise, the `reload()` builtin will not work correctly.)
         If a module named 'fullname' does not exist in
         `sys.modules`, the loader must create a new module object
         and add it to `sys.modules`.

         Note that the module object must be in `sys.modules`
         before the loader executes the module code. This is
         crucial because the module code may (directly or
         indirectly) import itself; adding it to `sys.modules`
         beforehand prevents unbounded recursion in the worst case
         and multiple loading in the best.

         If the load fails, the loader needs to remove any module it
         may have inserted into `sys.modules`. If the module was
         already in `sys.modules` then the loader should leave it
         alone.

       * The `__file__` attribute must be set. This must be a string,
         but it may be a dummy value, for example "<frozen>".
         The privilege of not having a `__file__` attribute at all
         is reserved for built-in modules.

       * The `__name__` attribute must be set. If one uses
         `imp.new_module()` then the attribute is set automatically.

       * If it's a package, the __path__ variable must be set.
         This must be a list, but may be empty if `__path__` has no
         further significance to the importer (more on this later).

       * The `__loader__` attribute must be set to the loader object.
         This is mostly for introspection and reloading, but can be
         used for importer-specific extras, for example getting data
         associated with an importer.

        The `__package__` attribute [8] must be set.

        If the module is a Python module (as opposed to a built-in
        module or a dynamically loaded extension), it should execute
        the module's code in the module's global name space
        (`module.__dict__`).

       [8] PEP 366: Main module explicit relative imports
           http://www.python.org/dev/peps/pep-0366/
    '''

    scope = ldr.scope.split('.')
    modpath = fqname.split('.')

    if scope != modpath[0:len(scope)]:
      raise AssertionError(
        "%s responsible for %s got request for %s" % (
          ldr.__class__.__name__,
          ldr.scope,
          fqname,
        )
      )

    if fqname in sys.modules:
      mod = sys.modules[fqname]
    else:
      mod = sys.modules.setdefault(fqname, types.ModuleType(fqname))

    mod.__loader__ = ldr

    fspath = ldr.path_to(fqname)

    mod.__file__ = str(fspath)

    if fs.is_package(fspath):
      mod.__path__ = [ldr.fspath]
      mod.__package__ = str(fqname)
    else:
      mod.__package__ = str(fqname.rpartition('.')[0])

    exec(fs.get_code(fspath), mod.__dict__)

    return mod