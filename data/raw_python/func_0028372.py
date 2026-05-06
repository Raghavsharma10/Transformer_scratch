async def get_or_create(cls, lang: str, *,
                            client_token: str = None,
                            mounts: Iterable[str] = None,
                            envs: Mapping[str, str] = None,
                            resources: Mapping[str, int] = None,
                            cluster_size: int = 1,
                            tag: str = None,
                            owner_access_key: str = None) -> 'Kernel':
        '''
        Get-or-creates a compute session.
        If *client_token* is ``None``, it creates a new compute session as long as
        the server has enough resources and your API key has remaining quota.
        If *client_token* is a valid string and there is an existing compute session
        with the same token and the same *lang*, then it returns the :class:`Kernel`
        instance representing the existing session.

        :param lang: The image name and tag for the compute session.
            Example: ``python:3.6-ubuntu``.
            Check out the full list of available images in your server using (TODO:
            new API).
        :param client_token: A client-side identifier to seamlessly reuse the compute
            session already created.
        :param mounts: The list of vfolder names that belongs to the currrent API
            access key.
        :param envs: The environment variables which always bypasses the jail policy.
        :param resources: The resource specification. (TODO: details)
        :param cluster_size: The number of containers in this compute session.
            Must be at least 1.
        :param tag: An optional string to annotate extra information.
        :param owner: An optional access key that owns the created session. (Only
            available to administrators)

        :returns: The :class:`Kernel` instance.
        '''
        if client_token:
            assert 4 <= len(client_token) <= 64, \
                   'Client session token should be 4 to 64 characters long.'
        else:
            client_token = uuid.uuid4().hex
        if mounts is None:
            mounts = []
        if resources is None:
            resources = {}
        mounts.extend(cls.session.config.vfolder_mounts)
        rqst = Request(cls.session, 'POST', '/kernel/create')
        rqst.set_json({
            'lang': lang,
            'tag': tag,
            'clientSessionToken': client_token,
            'config': {
                'mounts': mounts,
                'environ': envs,
                'clusterSize': cluster_size,
                'resources': resources,
            },
        })
        async with rqst.fetch() as resp:
            data = await resp.json()
            o = cls(data['kernelId'], owner_access_key)  # type: ignore
            o.created = data.get('created', True)     # True is for legacy
            o.service_ports = data.get('servicePorts', [])
            return o