def suggest(self, name, term, **kwargs):
        """Set suggestion options.

        :arg name: The name to use for the suggestions.
        :arg term: The term to suggest similar looking terms for.

        Additional keyword options:

        * ``field`` -- The field to base suggestions upon, defaults to _all

        Results will have a ``_suggestions`` property containing the
        suggestions for all terms.

        .. Note::

           Suggestions are only supported since Elasticsearch 0.90.

           Calling this multiple times will add multiple suggest clauses to
           the query.
        """
        return self._clone(next_step=('suggest', (name, term, kwargs)))