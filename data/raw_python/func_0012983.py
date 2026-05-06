def _bind(self, args, kwds):
    """Bind parameter values.  Returns a new Query object."""
    bindings = dict(kwds)
    for i, arg in enumerate(args):
      bindings[i + 1] = arg
    used = {}
    ancestor = self.ancestor
    if isinstance(ancestor, ParameterizedThing):
      ancestor = ancestor.resolve(bindings, used)
    filters = self.filters
    if filters is not None:
      filters = filters.resolve(bindings, used)
    unused = []
    for i in xrange(1, 1 + len(args)):
      if i not in used:
        unused.append(i)
    if unused:
      raise datastore_errors.BadArgumentError(
          'Positional arguments %s were given but not used.' %
          ', '.join(str(i) for i in unused))
    return self.__class__(kind=self.kind, ancestor=ancestor,
                          filters=filters, orders=self.orders,
                          app=self.app, namespace=self.namespace,
                          default_options=self.default_options,
                          projection=self.projection, group_by=self.group_by)