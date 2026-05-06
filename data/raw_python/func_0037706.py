def get_attribute(self, attribute):
        """
        :param attribute: requested attributes.
        :return: attribute value.
        :raise TgnError: if invalid attribute.
        """
        value = self.api.getAttribute(self.obj_ref(), attribute)
        # IXN returns '::ixNet::OK' for invalid attributes. We want error.
        if value == '::ixNet::OK':
            raise TgnError(self.ref + ' does not have attribute ' + attribute)
        return str(value)