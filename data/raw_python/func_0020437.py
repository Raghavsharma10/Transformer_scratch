def set_secrets(self, secrets):
        """
        :param secrets: dict, {(plugin type, plugin name, argument name): secret name}
            for example {('exit_plugins', 'koji_promote', 'koji_ssl_certs'): 'koji_ssl_certs', ...}
        """
        secret_set = False
        for (plugin, secret) in secrets.items():
            if not isinstance(plugin, tuple) or len(plugin) != 3:
                raise ValueError('got "%s" as secrets key, need 3-tuple' % plugin)
            if secret is not None:
                if isinstance(secret, list):
                    for secret_item in secret:
                        self.set_secret_for_plugin(secret_item, plugin=plugin)
                else:
                    self.set_secret_for_plugin(secret, plugin=plugin)
                secret_set = True

        if not secret_set:
            # remove references to secret if no secret was set
            if 'secrets' in self.template['spec']['strategy']['customStrategy']:
                del self.template['spec']['strategy']['customStrategy']['secrets']