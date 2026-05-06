def render_customizations(self):
        """
        Customize prod_inner for site specific customizations
        """

        disable_plugins = self.customize_conf.get('disable_plugins', [])
        if not disable_plugins:
            logger.debug("No site-specific plugins to disable")
        else:
            for plugin_dict in disable_plugins:
                try:
                    self.dj.remove_plugin(
                        plugin_dict['plugin_type'],
                        plugin_dict['plugin_name']
                    )
                    logger.debug(
                        "site-specific plugin disabled -> Type:{} Name:{}".format(
                            plugin_dict['plugin_type'],
                            plugin_dict['plugin_name']
                        )
                    )
                except KeyError:
                    # Malformed config
                    logger.debug("Invalid custom configuration found for disable_plugins")

        enable_plugins = self.customize_conf.get('enable_plugins', [])
        if not enable_plugins:
            logger.debug("No site-specific plugins to enable")
        else:
            for plugin_dict in enable_plugins:
                try:
                    self.dj.add_plugin(
                        plugin_dict['plugin_type'],
                        plugin_dict['plugin_name'],
                        plugin_dict['plugin_args']
                    )
                    logger.debug(
                        "site-specific plugin enabled -> Type:{} Name:{} Args: {}".format(
                            plugin_dict['plugin_type'],
                            plugin_dict['plugin_name'],
                            plugin_dict['plugin_args']
                        )
                    )
                except KeyError:
                    # Malformed config
                    logger.debug("Invalid custom configuration found for enable_plugins")