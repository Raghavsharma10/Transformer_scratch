def checkAndCreate(self, key, payload, domainId):
        """ Function checkAndCreate
        Check if a subnet exists and create it if not

        @param key: The targeted subnet
        @param payload: The targeted subnet description
        @param domainId: The domainId to be attached wiuth the subnet
        @return RETURN: The id of the subnet
        """
        if key not in self:
            self[key] = payload
        oid = self[key]['id']
        if not oid:
            return False
        #~ Ensure subnet contains the domain
        subnetDomainIds = []
        for domain in self[key]['domains']:
            subnetDomainIds.append(domain['id'])
        if domainId not in subnetDomainIds:
            subnetDomainIds.append(domainId)
            self[key]["domain_ids"] = subnetDomainIds
            if len(self[key]["domains"]) is not len(subnetDomainIds):
                return False
        return oid