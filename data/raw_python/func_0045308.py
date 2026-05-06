def reload(self):
        """
        Reloads current instance from DB store
        """
        self._load_data(self.objects.data().filter(key=self.key)[0][0], True)