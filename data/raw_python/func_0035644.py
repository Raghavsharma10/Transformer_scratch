def update(cls, session, record):
        """Update a record.

        Args:
            session (requests.sessions.Session): Authenticated session.
            record (helpscout.BaseModel): The record to
                be updated.

        Returns:
            helpscout.BaseModel: Freshly updated record.
        """
        cls._check_implements('update')
        data = record.to_api()
        del data['id']
        data['reload'] = True
        return cls(
            '/%s/%s.json' % (cls.__endpoint__, record.id),
            data=data,
            request_type=RequestPaginator.PUT,
            singleton=True,
            session=session,
        )