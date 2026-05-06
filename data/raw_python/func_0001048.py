def _extract(self, attribute) -> str:
        """Extract train information."""
        attr_data = self.journey.JourneyAttributeList.JourneyAttribute[
            self.attr_types.index(attribute)
        ].Attribute
        attr_variants = attr_data.xpath("AttributeVariant/@type")
        data = attr_data.AttributeVariant[attr_variants.index("NORMAL")].Text.pyval
        return str(data)