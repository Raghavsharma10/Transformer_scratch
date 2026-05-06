def apply_remote_config(self, namespace=None):
        """
        Applies all config values defined in consul's kv store to self.app.

        There is no guarantee that these values will not be overwritten later
        elsewhere.

        :param namespace: kv namespace/directory. Defaults to
                DEFAULT_KV_NAMESPACE
        :return: None
        """

        if namespace is None:
            namespace = "config/{service}/{environment}/".format(
                service=os.environ.get('SERVICE', 'generic_service'),
                environment=os.environ.get('ENVIRONMENT', 'generic_environment')
            )

        for k, v in iteritems(self.session.kv.find(namespace)):
            k = k.replace(namespace, '')
            try:
                self.app.config[k] = json.loads(v)
            except (TypeError, ValueError):
                self.app.logger.warning("Couldn't de-serialize {} to json, using raw value".format(v))
                self.app.config[k] = v

            msg = "Set {k}={v} from consul kv '{ns}'".format(
                k=k,
                v=v,
                ns=namespace,
            )
            self.app.logger.debug(msg)