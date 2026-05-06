def with_labels(self, selectable, base=None, **mapper_args):
        """Map a selectable directly, wrapping the 
        selectable in a subquery with labels.

        The class and its mapping are not cached and will
        be discarded once dereferenced (as of 0.6.6).

        :param selectable: an :func:`.expression.select` construct.
        :param base: a Python class which will be used as the
          base for the mapped class. If ``None``, the "base"
          argument specified by this :class:`.SQLSoup`
          instance's constructor will be used, which defaults to
          ``object``.
        :param mapper_args: Dictionary of arguments which will
          be passed directly to :func:`.orm.mapper`.

        """

        # TODO give meaningful aliases
        return self.map(
                    expression._clause_element_as_expr(selectable).
                            select(use_labels=True).
                            alias('foo'), base=base, **mapper_args)