def search(cls, session, queries):
        """Search for a customer given a domain.

        Args:
            session (requests.sessions.Session): Authenticated session.
            queries (helpscout.models.Domain or iter): The queries for the
                domain. If a ``Domain`` object is provided, it will simply be
                returned. Otherwise, a ``Domain`` object will be generated
                from the complex queries. In this case, the queries should
                conform to the interface in
                :func:`helpscout.domain.Domain.from_tuple`.

        Returns:
            RequestPaginator(output_type=helpscout.models.SearchCustomer):
                SearchCustomer iterator.
        """
        return super(Customers, cls).search(session, queries, SearchCustomer)