def child_get_property(self, child, property_name, value=None):
        """child_get_property(child, property_name, value=None)

        :param child:
            a widget which is a child of `self`
        :type child: :obj:`Gtk.Widget`

        :param property_name:
            the name of the property to get
        :type property_name: :obj:`str`

        :param value:
            Either :obj:`None` or a correctly initialized :obj:`GObject.Value`
        :type value: :obj:`GObject.Value` or :obj:`None`

        :returns: The Python value of the child property

        {{ docs }}
        """

        if value is None:
            prop = self.find_child_property(property_name)
            if prop is None:
                raise ValueError('Class "%s" does not contain child property "%s"' %
                                 (self, property_name))
            value = GObject.Value(prop.value_type)

        Gtk.Container.child_get_property(self, child, property_name, value)
        return value.get_value()