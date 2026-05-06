def get_value(self, property_name):
        """Returns the value associated to the passed property

        This public method is passed a specific property as a string
        and returns the value of that property. If the property is not
        found, None will be returned.

        :param property_name (str) The name of the property
        :return: (str) value for the passed property, or None.
        """
        log = logging.getLogger(self.cls_logger + '.get_value')
        if not isinstance(property_name, basestring):
            log.error('property_name arg is not a string, found type: {t}'.format(t=property_name.__class__.__name__))
            return None
        # Ensure a property with that name exists
        prop = self.get_property(property_name)
        if not prop:
            log.debug('Property name not found matching: {n}'.format(n=property_name))
            return None
        value = self.properties[prop]
        log.debug('Found value for property {n}: {v}'.format(n=property_name, v=value))
        return value