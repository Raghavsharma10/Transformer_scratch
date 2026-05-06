def get_context(app, package, module, fullname):
    """Return a dict for template rendering

    Variables:

      * :package: The top package
      * :module: the module
      * :fullname: package.module
      * :subpkgs: packages beneath module
      * :submods: modules beneath module
      * :classes: public classes in module
      * :allclasses: public and private classes in module
      * :exceptions: public exceptions in module
      * :allexceptions: public and private exceptions in module
      * :functions: public functions in module
      * :allfunctions: public and private functions in module
      * :data: public data in module
      * :alldata: public and private data in module
      * :members: dir(module)


    :param app: the sphinx app
    :type app: :class:`sphinx.application.Sphinx`
    :param package: the parent package name
    :type package: str
    :param module: the module name
    :type module: str
    :param fullname: package.module
    :type fullname: str
    :returns: a dict with variables for template rendering
    :rtype: :class:`dict`
    :raises: None
    """
    var = {'package': package,
           'module': module,
           'fullname': fullname}
    logger.debug('Creating context for: package %s, module %s, fullname %s', package, module, fullname)
    obj = import_name(app, fullname)
    if not obj:
        for k in ('subpkgs', 'submods', 'classes', 'allclasses',
                  'exceptions', 'allexceptions', 'functions', 'allfunctions',
                  'data', 'alldata', 'memebers'):
            var[k] = []
        return var

    var['subpkgs'] = get_subpackages(app, obj)
    var['submods'] = get_submodules(app, obj)
    var['classes'], var['allclasses'] = get_members(app, obj, 'class')
    var['exceptions'], var['allexceptions'] = get_members(app, obj, 'exception')
    var['functions'], var['allfunctions'] = get_members(app, obj, 'function')
    var['data'], var['alldata'] = get_members(app, obj, 'data')
    var['members'] = get_members(app, obj, 'members')
    logger.debug('Created context: %s', var)
    return var