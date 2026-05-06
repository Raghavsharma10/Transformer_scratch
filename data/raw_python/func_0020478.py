def render_customizations(self):
        """
        Customize template for site user specified customizations
        """
        disable_plugins = self.pt.customize_conf.get('disable_plugins', [])
        if not disable_plugins:
            logger.debug('No site-user specified plugins to disable')
        else:
            for plugin in disable_plugins:
                try:
                    self.pt.remove_plugin(plugin['plugin_type'], plugin['plugin_name'],
                                          'disabled at user request')
                except KeyError:
                    # Malformed config
                    logger.info('Invalid custom configuration found for disable_plugins')

        enable_plugins = self.pt.customize_conf.get('enable_plugins', [])
        if not enable_plugins:
            logger.debug('No site-user specified plugins to enable"')
        else:
            for plugin in enable_plugins:
                try:
                    msg = 'enabled at user request'
                    self.pt.add_plugin(plugin['plugin_type'], plugin['plugin_name'],
                                       plugin['plugin_args'], msg)
                except KeyError:
                    # Malformed config
                    logger.info('Invalid custom configuration found for enable_plugins')