def importPuppetClasses(self, smartProxyId):
        """ Function importPuppetClasses
        Force the reload of puppet classes

        @param smartProxyId: smartProxy Id
        @return RETURN: the API result
        """
        return self.api.create('{}/{}/import_puppetclasses'
                               .format(self.objName, smartProxyId), '{}')