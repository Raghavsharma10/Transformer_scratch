def removeDomain(self, subnetId, domainId):
        """ Function removeDomain
        Delete a domain from a subnet

        @param subnetId: The subnet Id
        @param domainId: The domainId to be attached wiuth the subnet
        @return RETURN: boolean
        """
        subnetDomainIds = []
        for domain in self[subnetId]['domains']:
            subnetDomainIds.append(domain['id'])
        subnetDomainIds.remove(domainId)
        self[subnetId]["domain_ids"] = subnetDomainIds
        return len(self[subnetId]["domains"]) is len(subnetDomainIds)