def list(cls, session, endpoint_override=None, data=None):
        """Return records in a mailbox.

        Args:
            session (requests.sessions.Session): Authenticated session.
            endpoint_override (str, optional): Override the default
                endpoint using this.
            data (dict, optional): Data to provide as request parameters.

        Returns:
            RequestPaginator(output_type=helpscout.BaseModel): Results
                iterator.
        """
        cls._check_implements('list')
        return cls(
            endpoint_override or '/%s.json' % cls.__endpoint__,
            data=data,
            session=session,
        )