def checkAndCreate(self, key, payload):
        """ Function checkAndCreate
        Check if an object exists and create it if not

        @param key: The targeted object
        @param payload: The targeted object description
        @return RETURN: The id of the object
        """
        if key not in self:
            if 'templates' in payload:
                templates = payload.pop('templates')
            self[key] = payload
            self.reload()
        return self[key]['id']