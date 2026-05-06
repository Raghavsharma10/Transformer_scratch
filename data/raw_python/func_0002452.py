def first_or_new(self, _attributes=None, **attributes):
        """
        Get the first related model record matching the attributes or instantiate it.

        :param attributes:  The attributes
        :type attributes: dict

        :rtype: Model
        """
        if _attributes is not None:
            attributes.update(_attributes)

        instance = self.where(attributes).first()

        if instance is None:
            instance = self._related.new_instance()
            instance.set_attribute(self.get_plain_foreign_key(), self.get_parent_key())

        return instance