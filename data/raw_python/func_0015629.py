def style_get_property(self, property_name, value=None):
        """style_get_property(property_name, value=None)

        :param property_name:
            the name of a style property
        :type property_name: :obj:`str`

        :param value:
            Either :obj:`None` or a correctly initialized :obj:`GObject.Value`
        :type value: :obj:`GObject.Value` or :obj:`None`

        :returns: The Python value of the style property

        {{ docs }}
        """

        if value is None:
            prop = self.find_style_property(property_name)
            if prop is None:
                raise ValueError('Class "%s" does not contain style property "%s"' %
                                 (self, property_name))
            value = GObject.Value(prop.value_type)

        Gtk.Widget.style_get_property(self, property_name, value)
        return value.get_value()