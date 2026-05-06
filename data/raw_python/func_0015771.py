def demote(self, amount_, *queries, **kw):
        """
        Returns a new S instance with boosting query and demotion.

        You can demote documents that match query criteria::

            q = (S().query(title='trucks')
                    .demote(0.5, description__match='gross'))

            q = (S().query(title='trucks')
                    .demote(0.5, Q(description__match='gross')))

        This is implemented using the boosting query in
        Elasticsearch. Anything you specify with ``.query()`` goes
        into the positive section. The negative query and negative
        boost portions are specified as the first and second arguments
        to ``.demote()``.

        .. Note::

           Calling this again will overwrite previous ``.demote()``
           calls.

        """
        q = Q()
        for query in queries:
            q += query
        q += Q(**kw)

        return self._clone(next_step=('demote', (amount_, q)))