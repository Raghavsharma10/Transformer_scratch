def enhance(self):
        """ Function enhance
        Enhance the object with new item or enhanced items
        """
        self.update({'parameters':
                     SubDict(self.api, self.objName,
                             self.payloadObj, self.key,
                             SubItemParameter)})
        self.update({'interfaces':
                     SubDict(self.api, self.objName,
                             self.payloadObj, self.key,
                             SubItemInterface)})
        self.update({'subnets':
                    SubDict(self.api, self.objName,
                            self.payloadObj, self.key,
                            SubItemSubnet)})