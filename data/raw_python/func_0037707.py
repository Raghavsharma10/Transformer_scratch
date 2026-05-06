def get_list_attribute(self, attribute):
        """
        :return: attribute value as Python list.
        """
        list_attribute = self.api.getListAttribute(self.obj_ref(), attribute)
        # IXN returns '::ixNet::OK' for invalid attributes. We want error.
        if list_attribute == ['::ixNet::OK']:
            raise TgnError(self.ref + ' does not have attribute ' + attribute)
        return list_attribute