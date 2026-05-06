def analyze(self):
    """Return a list giving the parameters required by a query."""
    class MockBindings(dict):

      def __contains__(self, key):
        self[key] = None
        return True
    bindings = MockBindings()
    used = {}
    ancestor = self.ancestor
    if isinstance(ancestor, ParameterizedThing):
      ancestor = ancestor.resolve(bindings, used)
    filters = self.filters
    if filters is not None:
      filters = filters.resolve(bindings, used)
    return sorted(used)