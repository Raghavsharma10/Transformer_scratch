def checkAndCreate(self, key, payload, osIds):
        """ Function checkAndCreate
        Check if an architectures exists and create it if not

        @param key: The targeted architectures
        @param payload: The targeted architectures description
        @param osIds: The list of os ids liked with this architecture
        @return RETURN: The id of the object
        """
        if key not in self:
            self[key] = payload
        oid = self[key]['id']
        if not oid:
            return False
        #~ To be sure the OS list is good, we ensure our os are in the list
        for os in self[key]['operatingsystems']:
            osIds.add(os['id'])
        self[key]["operatingsystem_ids"] = list(osIds)
        if (len(self[key]['operatingsystems']) is not len(osIds)):
            return False
        return oid