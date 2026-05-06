def _perform_version(self, version):
        """Inner method for version upgrade.

        Not intended for standalone use. This method performs the actual
        version upgrade with all the pre, post operations and addons upgrades.

        :param version: The migration version to upgrade to
        :type version: Instance of Version class
        """
        if version.is_noop():
            self.log(u'version {} is a noop'.format(version.number))
        else:
            self.log(u'execute base pre-operations')
            for operation in version.pre_operations():
                operation.execute(self.log)
            if self.config.mode:
                self.log(u'execute %s pre-operations' % self.config.mode)
                for operation in version.pre_operations(mode=self.config.mode):
                    operation.execute(self.log)

            self.perform_addons()

            self.log(u'execute base post-operations')
            for operation in version.post_operations():
                operation.execute(self.log)
            if self.config.mode:
                self.log(u'execute %s post-operations' % self.config.mode)
                for operation in version.post_operations(self.config.mode):
                    operation.execute(self.log)