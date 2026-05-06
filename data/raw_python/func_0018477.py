def join(self, left, right, onclause=None, isouter=False, 
                base=None, **mapper_args):
        """Create an :func:`.expression.join` and map to it.

        The class and its mapping are not cached and will
        be discarded once dereferenced (as of 0.6.6).

        :param left: a mapped class or table object.
        :param right: a mapped class or table object.
        :param onclause: optional "ON" clause construct..
        :param isouter: if True, the join will be an OUTER join.
        :param base: a Python class which will be used as the
          base for the mapped class. If ``None``, the "base"
          argument specified by this :class:`.SQLSoup`
          instance's constructor will be used, which defaults to
          ``object``.
        :param mapper_args: Dictionary of arguments which will
          be passed directly to :func:`.orm.mapper`.

        """

        j = join(left, right, onclause=onclause, isouter=isouter)
        return self.map(j, base=base, **mapper_args)