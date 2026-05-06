def enhance(self):
        """ Function enhance
        Enhance the object with new item or enhanced items
        """
        self.update({'images':
                     SubDict(self.api, self.objName,
                             self.payloadObj, self.key,
                             SubItemImages)})