def _get_publish(self):
        """
        Find this publish on remote
        """
        publishes = self._get_publishes(self.client)
        for publish in publishes:
            if publish['Distribution'] == self.distribution and \
                    publish['Prefix'].replace("/", "_") == (self.prefix or '.') and \
                    publish['Storage'] == self.storage:
                return publish
        raise NoSuchPublish("Publish %s (%s) does not exist" % (self.name, self.storage or "local"))