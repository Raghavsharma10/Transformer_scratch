def search(cls, session, queries, out_type):
        """Search for a record given a domain.

        Args:
            session (requests.sessions.Session): Authenticated session.
            queries (helpscout.models.Domain or iter): The queries for the
                domain. If a ``Domain`` object is provided, it will simply be
                returned. Otherwise, a ``Domain`` object will be generated
                from the complex queries. In this case, the queries should
                conform to the interface in
                :func:`helpscout.domain.Domain.from_tuple`.
            out_type (helpscout.BaseModel): The type of record to output. This
                should be provided by child classes, by calling super.

        Returns:
            RequestPaginator(output_type=helpscout.BaseModel): Results
                iterator of the ``out_type`` that is defined.
        """
        cls._check_implements('search')
        domain = cls.get_search_domain(queries)
        return cls(
            '/search/%s.json' % cls.__endpoint__,
            data={'query': str(domain)},
            session=session,
            out_type=out_type,
        )