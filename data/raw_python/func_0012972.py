def filter(self, *args):
    """Return a new Query with additional filter(s) applied."""
    if not args:
      return self
    preds = []
    f = self.filters
    if f:
      preds.append(f)
    for arg in args:
      if not isinstance(arg, Node):
        raise TypeError('Cannot filter a non-Node argument; received %r' % arg)
      preds.append(arg)
    if not preds:
      pred = None
    elif len(preds) == 1:
      pred = preds[0]
    else:
      pred = ConjunctionNode(*preds)
    return self.__class__(kind=self.kind, ancestor=self.ancestor,
                          filters=pred, orders=self.orders,
                          app=self.app, namespace=self.namespace,
                          default_options=self.default_options,
                          projection=self.projection, group_by=self.group_by)