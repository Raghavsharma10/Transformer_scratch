def create_source(self, datapusher=True):
        """
        Populate ckan directory from preloaded image and copy
        who.ini and schema.xml info conf directory
        """
        task.create_source(self.target, self._preload_image(), datapusher)