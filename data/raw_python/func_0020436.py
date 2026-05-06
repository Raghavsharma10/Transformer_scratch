def set_secret_for_plugin(self, secret, plugin=None, mount_path=None):
        """
        Sets secret for plugin, if no plugin specified
        it will also set general secret

        :param secret: str, secret name
        :param plugin: tuple, (plugin type, plugin name, argument name)
        :param mount_path: str, mount path of secret
        """
        has_plugin_conf = False
        if plugin is not None:
            has_plugin_conf = self.dj.dock_json_has_plugin_conf(plugin[0],
                                                                plugin[1])
        if 'secrets' in self.template['spec']['strategy']['customStrategy']:
            if not plugin or has_plugin_conf:

                custom = self.template['spec']['strategy']['customStrategy']
                if mount_path:
                    secret_path = mount_path
                else:
                    secret_path = os.path.join(SECRETS_PATH, secret)

                logger.info("Configuring %s secret at %s", secret, secret_path)
                existing = [secret_mount for secret_mount in custom['secrets']
                            if secret_mount['secretSource']['name'] == secret]
                if existing:
                    logger.debug("secret %s already set", secret)
                else:
                    custom['secrets'].append({
                        'secretSource': {
                            'name': secret,
                        },
                        'mountPath': secret_path,
                    })

                # there's no need to set args if no plugin secret specified
                # this is used in tag_and_push plugin, as it sets secret path
                # for each registry separately
                if plugin and plugin[2] is not None:
                    self.dj.dock_json_set_arg(*(plugin + (secret_path,)))
            else:
                logger.debug("not setting secret for unused plugin %s",
                             plugin[1])