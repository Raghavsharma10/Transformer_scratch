def create(cls, object_type=None, object_uuid=None, **kwargs):
        """Create a new record identifier.

        :param object_type: The object type. (Default: ``None``)
        :param object_uuid: The object UUID. (Default: ``None``)
        """
        assert 'pid_value' in kwargs

        kwargs.setdefault('status', cls.default_status)
        if object_type and object_uuid:
            kwargs['status'] = PIDStatus.REGISTERED

        return super(OAIIDProvider, cls).create(
            object_type=object_type, object_uuid=object_uuid, **kwargs)