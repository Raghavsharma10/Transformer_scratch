def _parse_property_list(prop, value):
        """Parse a list property and return a list of the results."""
        attributes = []
        for v in value:
            try:
                attributes.append(
                    prop.prop.instance_class.from_api(**v),
                )
            except AttributeError:
                attributes.append(v)
        return attributes