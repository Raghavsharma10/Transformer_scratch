def setorigin(self):
        """Set the 'origin' remote to the upstream url that we trust."""
        try:
            origin = self.repo.remotes.origin
            if origin.url != self.origin_url:
                log.debug('[%s] Changing origin url. Old: %s New: %s',
                          self.name, origin.url, self.origin_url)
                origin.config_writer.set('url', self.origin_url)
        except AttributeError:
            origin = self.repo.create_remote('origin', self.origin_url)
            log.debug('[%s] Created remote "origin" with URL: %s',
                      self.name, origin.url)