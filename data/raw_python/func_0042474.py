def add_to_class(cls, name, value):
        """
        Django 1.10 doesn't allow us to modify `_base_manager` attr:
        https://docs.djangoproject.com/en/1.10/topics/db/managers/#django.db.models.Model._base_manager
        """
        if name == '_base_manager':
            if not value.name:
                value.name = name
            value.model = cls
            setattr(cls._meta, 'base_manager_name', 'normal')
        else:
            super(VersionModelMeta, cls).add_to_class(name, value)