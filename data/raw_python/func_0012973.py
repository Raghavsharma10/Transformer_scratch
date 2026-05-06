def order(self, *args):
    """Return a new Query with additional sort order(s) applied."""
    # q.order(Employee.name, -Employee.age)
    if not args:
      return self
    orders = []
    o = self.orders
    if o:
      orders.append(o)
    for arg in args:
      if isinstance(arg, model.Property):
        orders.append(datastore_query.PropertyOrder(arg._name, _ASC))
      elif isinstance(arg, datastore_query.Order):
        orders.append(arg)
      else:
        raise TypeError('order() expects a Property or query Order; '
                        'received %r' % arg)
    if not orders:
      orders = None
    elif len(orders) == 1:
      orders = orders[0]
    else:
      orders = datastore_query.CompositeOrder(orders)
    return self.__class__(kind=self.kind, ancestor=self.ancestor,
                          filters=self.filters, orders=orders,
                          app=self.app, namespace=self.namespace,
                          default_options=self.default_options,
                          projection=self.projection, group_by=self.group_by)