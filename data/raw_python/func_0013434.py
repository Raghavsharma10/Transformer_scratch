def serializeAttributes(self, msg):
        """
        Sets the attrbutes of a message during serialization.
        """
        attributes = self.getAttributes()
        for key in attributes:
            protocol.setAttribute(
                msg.attributes.attr[key].values, attributes[key])
        return msg