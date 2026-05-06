def style_node(self, additional_style_attrib=None):
        """
        generate a style node (for automatic-styles)

        could specify additional attributes such as
        'style:parent-style-name' or 'style:list-style-name'

        """
        style_attrib = {"style:name": self.name, "style:family": self.FAMILY}
        if additional_style_attrib:
            style_attrib.update(additional_style_attrib)
        if self.PARENT_STYLE_DICT:
            style_attrib.update(self.PARENT_STYLE_DICT)

        node = el("style:style", attrib=style_attrib)
        props = sub_el(node, self.STYLE_PROP, attrib=self.styles)
        return node