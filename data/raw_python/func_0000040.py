def switches(self):
        """
        List of all switches currently registered.
        """
        results = [
            switch for name, switch in self.storage.iteritems()
            if name.startswith(self.__joined_namespace)
        ]

        return results