def get(cls, session, record_id, endpoint_override=None):
        """Return a specific record.

        Args:
            session (requests.sessions.Session): Authenticated session.
            record_id (int): The ID of the record to get.
            endpoint_override (str, optional): Override the default
                endpoint using this.

        Returns:
            helpscout.BaseModel: A record singleton, if existing. Otherwise
                ``None``.
        """
        cls._check_implements('get')
        try:
            return cls(
                endpoint_override or '/%s/%d.json' % (
                    cls.__endpoint__, record_id,
                ),
                singleton=True,
                session=session,
            )
        except HelpScoutRemoteException as e:
            if e.status_code == 404:
                return None
            else:
                raise