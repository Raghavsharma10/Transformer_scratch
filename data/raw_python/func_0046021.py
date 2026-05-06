def detected_releasers(cls, config):
        """
        Returns all of the releasers that are compatible with the project.
        """

        def get_config(releaser):
            if config:
                return config.get(releaser.config_name(), {})

            return {}

        releasers = []

        for releaser_cls in cls.releasers():
            releaser_config = get_config(releaser_cls)

            if releaser_config.get('disabled', False):
                continue

            if releaser_cls.detect():
                logger.info('Enabled Releaser: {}'.format(releaser_cls.name))
                releasers.append(releaser_cls(releaser_config))

        return releasers