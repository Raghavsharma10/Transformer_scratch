def setAttributesJson(self, attributesJson):
        """
        Sets the attributes dictionary from a JSON string.
        """
        try:
            self._attributes = json.loads(attributesJson)
        except:
            raise exceptions.InvalidJsonException(attributesJson)
        return self