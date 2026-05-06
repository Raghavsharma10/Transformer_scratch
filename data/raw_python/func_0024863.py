def create(self):
        """
        Create an instance of the Parking Planning Service with the
        typical starting settings.
        """
        self.service.create()
        os.environ[self.__module__ + '.uri'] = self.service.settings.data['url']
        os.environ[self.__module__ + '.zone_id'] = self.get_predix_zone_id()