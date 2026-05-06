def update_style(self, mapping):
        """Use to update fill-color"""
        default = {
            "presentation:background-visible": "true",
            "presentation:background-objects-visible": "true",
            "draw:fill": "solid",
            "draw:fill-color": "#772953",
            "draw:fill-image-width": "0cm",
            "draw:fill-image-height": "0cm",
            "presentation:display-footer": "true",
            "presentation:display-page-number": "false",
            "presentation:display-date-time": "true",
        }
        default.update(mapping)
        style = PageStyle(**default)
        node = style.style_node()
        # add style to automatic-style
        self.preso._auto_styles.append(node)
        # update page style-name
        # found in ._page
        self._page.set(ns("draw", "style-name"), node.attrib[ns("style", "name")])