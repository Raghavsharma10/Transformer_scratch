def unit(self):
        """The string that is this instances prefix unit name in agreement
with this instance value (singular or plural). Following the
convention that only 1 is singular. This will always be the singular
form when :attr:`bitmath.format_plural` is ``False`` (default value).

For example:

   >>> KiB(1).unit == 'KiB'
   >>> Byte(0).unit == 'Bytes'
   >>> Byte(1).unit == 'Byte'
   >>> Byte(1.1).unit == 'Bytes'
   >>> Gb(2).unit == 'Gbs'

        """
        global format_plural

        if self.prefix_value == 1:
            # If it's a '1', return it singular, no matter what
            return self._name_singular
        elif format_plural:
            # Pluralization requested
            return self._name_plural
        else:
            # Pluralization NOT requested, and the value is not 1
            return self._name_singular