def create(cls, session, record, endpoint_override=None, out_type=None,
               **add_params):
        """Create an object on HelpScout.

        Args:
            session (requests.sessions.Session): Authenticated session.
            record (helpscout.BaseModel): The record to be created.
            endpoint_override (str, optional): Override the default
                endpoint using this.
            out_type (helpscout.BaseModel, optional): The type of record to
                output. This should be provided by child classes, by calling
                super.
            **add_params (mixed): Add these to the request parameters.

        Returns:
            helpscout.models.BaseModel: Newly created record. Will be of the
        """
        cls._check_implements('create')
        data = record.to_api()
        params = {
            'reload': True,
        }
        params.update(**add_params)
        data.update(params)
        return cls(
            endpoint_override or '/%s.json' % cls.__endpoint__,
            data=data,
            request_type=RequestPaginator.POST,
            singleton=True,
            session=session,
            out_type=out_type,
        )