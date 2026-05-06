def delete(cls, session, record, endpoint_override=None, out_type=None):
        """Delete a record.

        Args:
            session (requests.sessions.Session): Authenticated session.
            record (helpscout.BaseModel): The record to be deleted.
            endpoint_override (str, optional): Override the default
                endpoint using this.
            out_type (helpscout.BaseModel, optional): The type of record to
                output. This should be provided by child classes, by calling
                super.

        Returns:
            NoneType: Nothing.
        """
        cls._check_implements('delete')
        return cls(
            endpoint_override or '/%s/%s.json' % (
                cls.__endpoint__, record.id,
            ),
            request_type=RequestPaginator.DELETE,
            singleton=True,
            session=session,
            out_type=out_type,
        )