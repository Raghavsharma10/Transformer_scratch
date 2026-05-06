def instantiate_client(self, config):
        """
        :param config: The config object.
        :type config: dict
        :return: The instantiated class.
        :rtype: :class:`revision.client.Client`
        """
        modules = config.module.split('.')
        class_name = modules.pop()
        module_path = '.'.join(modules)

        client_instance = getattr(
            __import__(module_path, {}, {}, ['']),
            class_name
        )()

        client_instance.add_config(config)

        return client_instance