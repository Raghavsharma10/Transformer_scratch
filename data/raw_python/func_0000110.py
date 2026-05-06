def checkAndCreate(self, key, payload):
        """ Function checkAndCreate
        Check if an object exists and create it if not

        @param key: The targeted object
        @param payload: The targeted object description
        @return RETURN: The id of the object
        """
        if key not in self:
            self[key] = payload
        return self[key]['id']