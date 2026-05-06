def add_colors(self, system, colors):
        """Add color definition to a given color system.

        You may add to already existing color system. Previously existing color
        definitions of the same (normalized) name will be overwritten,
        regardless of the color system.

        Args:
          system (string): The color system the colors should be added to
            (e.g. ``"en"``).
          color_definitions (iterable of tuples): Color name / sRGB value pairs
            (e.g.  ``[("white", "ffffff"), ("red", "ff0000")]``)

        Examples:
          >>> color_definitions = {"greenish": "336633", "blueish": "334466"}
          >>> tint_registry = TintRegistry()
          >>> tint_registry.add_colors("vague", color_definitions.iteritems())

        """

        if system not in self._colors_by_system_hex:
            self._colors_by_system_hex[system] = {}
            self._colors_by_system_lab[system] = []

        for color_name, hex_code in colors:
            hex_code = hex_code.lower().strip().strip("#")
            color_name = color_name.lower().strip()
            if not isinstance(color_name, unicode):
                color_name = unicode(color_name, "utf-8")

            self._colors_by_system_hex[system][hex_code] = color_name
            self._colors_by_system_lab[system].append((_hex_to_lab(hex_code), color_name))
            self._hex_by_color[_normalize(color_name)] = hex_code