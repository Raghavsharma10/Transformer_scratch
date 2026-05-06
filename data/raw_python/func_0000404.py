def checkAndCreate(self, key, payload,
                       hostgroupConf,
                       hostgroupParent,
                       puppetClassesId):
        """ Function checkAndCreate
        check And Create procedure for an hostgroup
        - check the hostgroup is not existing
        - create the hostgroup
        - Add puppet classes from puppetClassesId
        - Add params from hostgroupConf

        @param key: The hostgroup name or ID
        @param payload: The description of the hostgroup
        @param hostgroupConf: The configuration of the host group from the
                              foreman.conf
        @param hostgroupParent: The id of the parent hostgroup
        @param puppetClassesId: The dict of puppet classes ids in foreman
        @return RETURN: The ItemHostsGroup object of an host
        """
        if key not in self:
            self[key] = payload
        oid = self[key]['id']
        if not oid:
            return False

        # Create Hostgroup classes
        if 'classes' in hostgroupConf.keys():
            classList = list()
            for c in hostgroupConf['classes']:
                classList.append(puppetClassesId[c])
            if not self[key].checkAndCreateClasses(classList):
                print("Failed in classes")
                return False

        # Set params
        if 'params' in hostgroupConf.keys():
            if not self[key].checkAndCreateParams(hostgroupConf['params']):
                print("Failed in params")
                return False

        return oid