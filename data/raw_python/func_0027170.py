def get_min_value(self):
        """ Get the minimum value """

        value = self.get_default_value()

        if self.attribute_type is str:
            min_value = value[:self.min_length - 1]

        elif self.attribute_type is int:
            min_value = self.min_length - 1

        else:
            raise TypeError('Attribute %s can not have a minimum value' % self.local_name)

        return min_value