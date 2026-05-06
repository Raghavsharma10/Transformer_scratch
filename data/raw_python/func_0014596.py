def ManagerMock(manager, *return_value):
    """
    Set the results to two items:

    >>> objects = ManagerMock(Post.objects, 'queryset', 'result')
    >>> assert objects.filter() == objects.all()

    Force an exception:

    >>> objects = ManagerMock(Post.objects, Exception())

    See QuerySetMock for more about how this works.
    """

    def make_get_query_set(self, model):
        def _get(*a, **k):
            return QuerySetMock(model, *return_value)
        return _get

    actual_model = getattr(manager, 'model', None)
    if actual_model:
        model = mock.MagicMock(spec=actual_model())
    else:
        model = mock.MagicMock()

    m = SharedMock()
    m.model = model
    m.get_query_set = make_get_query_set(m, actual_model)
    m.get = m.get_query_set().get
    m.count = m.get_query_set().count
    m.exists = m.get_query_set().exists
    m.__iter__ = m.get_query_set().__iter__
    m.__getitem__ = m.get_query_set().__getitem__
    return m