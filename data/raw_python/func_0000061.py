def enhance(self):
        """ Function enhance
        Enhance the object with new item or enhanced items
        """
        self.update({'os_default_templates':
                     SubDict(self.api, self.objName,
                             self.payloadObj, self.key,
                             SubItemOsDefaultTemplate)})
        self.update({'config_templates':
                     SubDict(self.api, self.objName,
                             self.payloadObj, self.key,
                             SubItemConfigTemplate)})
        self.update({'ptables':
                     SubDict(self.api, self.objName,
                             self.payloadObj, self.key,
                             SubItemPTable)})
        self.update({'media':
                     SubDict(self.api, self.objName,
                             self.payloadObj, self.key,
                             SubItemMedia)})
        self.update({'architectures':
                     SubDict(self.api, self.objName,
                             self.payloadObj, self.key,
                             SubItemArchitecture)})