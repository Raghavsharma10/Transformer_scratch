def find_module(fdr, fqname, path = None):
    '''Find a loader for module or package `fqname`.

       This method will be called with the fully qualified name
       of the module.  If the finder is installed on `sys.meta_path`,
       it will receive a second argument, which is `None` for
       a top-level module, or `package.__path__` for submodules
       or subpackages [5].
       It should return a loader object if the module was found, or
       `None` if it wasn't.  If `find_module()` raises an exception,
       it will be propagated to the caller, aborting the import.

       [5] The path argument to `finder.find_module()` is there
           because the `pkg.__path__` variable may be needed
           at this point.  It may either come from the actual
           parent module or be supplied by `imp.find_module()`
           or the proposed `imp.get_loader()` function.
    '''
    if fqname in fdr.aliases:
      return Loader(fqname, fdr.aliases[fqname])
    return None