def first_or_create(self, _attributes=None, **attributes):
        """
        Get the first related record matching the attributes or create it.

        :param attributes:  The attributes
        :type attributes: dict

        :rtype: Model
        """
        if _attributes is not None:
            attributes.update(_attributes)

        instance = self.where(attributes).first()

        if instance is None:
            instance = self.create(**attributes)

        return instance