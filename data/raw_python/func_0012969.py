def _fix_namespace(self):
    """Internal helper to fix the namespace.

    This is called to ensure that for queries without an explicit
    namespace, the namespace used by async calls is the one in effect
    at the time the async call is made, not the one in effect when the
    the request is actually generated.
    """
    if self.namespace is not None:
      return self
    namespace = namespace_manager.get_namespace()
    return self.__class__(kind=self.kind, ancestor=self.ancestor,
                          filters=self.filters, orders=self.orders,
                          app=self.app, namespace=namespace,
                          default_options=self.default_options,
                          projection=self.projection, group_by=self.group_by)