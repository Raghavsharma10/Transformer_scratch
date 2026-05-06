def _attribute_definition(self, attrid):
        """ Returns the attribute definition dict of a given attribute
        ID, can be the name or the integer ID """
        attrs = self._schema["attributes"]

        try:
            # Make a new dict to avoid side effects
            return dict(attrs[attrid])
        except KeyError:
            attr_names = self._schema["attribute_names"]
            attrdef = attrs.get(attr_names.get(str(attrid).lower()))

            if not attrdef:
                return None
            else:
                return dict(attrdef)