def filter(self, *filters, **kw):
        """
        Return a new S instance with filter args combined with
        existing set with AND.

        :arg filters: this will be instances of F
        :arg kw: this will be in the form of ``field__action=value``

        Examples:

        >>> s = S().filter(foo='bar')
        >>> s = S().filter(F(foo='bar'))
        >>> s = S().filter(foo='bar', bat='baz')
        >>> s = S().filter(foo='bar').filter(bat='baz')

        By default, everything is combined using AND. If you provide
        multiple filters in a single filter call, those are ANDed
        together. If you provide multiple filters in multiple filter
        calls, those are ANDed together.

        If you want something different, use the F class which supports
        ``&`` (and), ``|`` (or) and ``~`` (not) operators. Then call
        filter once with the resulting F instance.

        See the documentation on :py:class:`elasticutils.F` for more
        details on composing filters with F.

        See the documentation on :py:class:`elasticutils.S` for more
        details on adding support for new filter types.

        """
        items = kw.items()
        if six.PY3:
            items = list(items)
        return self._clone(
            next_step=('filter', list(filters) + items))