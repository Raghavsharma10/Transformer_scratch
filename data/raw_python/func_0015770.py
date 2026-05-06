def boost(self, **kw):
        """
        Return a new S instance with field boosts.

        ElasticUtils allows you to specify query-time field boosts
        with ``.boost()``. It takes a set of arguments where the keys
        are either field names or field name + ``__`` + field action.

        Examples::

            q = (S().query(title='taco trucks',
                           description__match='awesome')
                    .boost(title=4.0, description__match=2.0))


        If the key is a field name, then the boost will apply to all
        query bits that have that field name. For example::

            q = (S().query(title='trucks',
                           title__prefix='trucks',
                           title__fuzzy='trucks')
                    .boost(title=4.0))


        applies a 4.0 boost to all three query bits because all three
        query bits are for the title field name.

        If the key is a field name and field action, then the boost
        will apply only to that field name and field action. For
        example::

            q = (S().query(title='trucks',
                           title__prefix='trucks',
                           title__fuzzy='trucks')
                    .boost(title__prefix=4.0))


        will only apply the 4.0 boost to title__prefix.

        Boosts are relative to one another and all boosts default to
        1.0.

        For example, if you had::

            qs = (S().boost(title=4.0, summary=2.0)
                     .query(title__match=value,
                            summary__match=value,
                            content__match=value,
                            should=True))


        ``title__match`` would be boosted twice as much as
        ``summary__match`` and ``summary__match`` twice as much as
        ``content__match``.

        """
        new = self._clone()
        new.field_boosts.update(kw)
        return new