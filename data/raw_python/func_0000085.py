def enhance(self):
        """ Function enhance
        Enhance the object with new item or enhanced items
        """
        self.update({'puppetclasses':
                     SubDict(self.api, self.objName,
                             self.payloadObj, self.key,
                             SubItemPuppetClasses)})
        self.update({'parameters':
                     SubDict(self.api, self.objName,
                             self.payloadObj, self.key,
                             SubItemParameter)})
        self.update({'smart_class_parameters':
                    SubDict(self.api, self.objName,
                            self.payloadObj, self.key,
                            ItemSmartClassParameter)})