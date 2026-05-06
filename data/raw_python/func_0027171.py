def get_max_value(self):
        """ Get the maximum value """

        value = self.get_default_value()

        if self.attribute_type is str:
            max_value = value.ljust(self.max_length + 1, 'a')

        elif self.attribute_type is int:
            max_value = self.max_length + 1

        else:
            raise TypeError('Attribute %s can not have a maximum value' % self.local_name)

        return max_value