def release_assets(self, release):
        """Assets for a given release
        """
        release = self.as_id(release)
        return self.get_list(url='%s/%s/assets' % (self, release))