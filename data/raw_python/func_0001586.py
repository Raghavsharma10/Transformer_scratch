def save(self):
        """
        Save the config with the cleaned data,\
        update the last modified date so\
        the config is reloaded on other process/nodes.\
        Reload the config so it can be called right away.
        """
        assert self.__class__ in conf.config._registry,\
            '%(class_name)s is not registered' % {
                'class_name': self.__class__.__name__
            }

        ConfigModel = apps.get_model('djconfig.Config')

        for field_name, value in self.cleaned_data.items():
            value = utils.serialize(
                value=value,
                field=self.fields.get(field_name, None))
            # TODO: use update_or_create
            count = (ConfigModel.objects
                .filter(key=field_name)
                .update(value=value))
            if not count:
                ConfigModel.objects.create(
                    key=field_name,
                    value=value)

        count = (ConfigModel.objects
            .filter(key='_updated_at')
            .update(value=str(timezone.now())))
        if not count:
            ConfigModel.objects.create(
                key='_updated_at',
                value=str(timezone.now()))

        conf.config._reload()