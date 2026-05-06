def find_nearest(self, hex_code, system, filter_set=None):
        """Find a color name that's most similar to a given sRGB hex code.

        In normalization terms, this method implements "normalize an arbitrary sRGB value
        to a well-defined color name".

        Args:
          system (string): The color system. Currently, `"en"`` is the only default
            system.
          filter_set (iterable of string, optional): Limits the output choices
            to fewer color names. The names (e.g. ``["black", "white"]``) must be
            present in the given system.
            If omitted, all color names of the system are considered. Defaults to None.

        Returns:
          A named tuple with the members `color_name` and `distance`.

        Raises:
          ValueError: If argument `system` is not a registered color system.

        Examples:
          >>> tint_registry = TintRegistry()
          >>> tint_registry.find_nearest("54e6e4", system="en")
          FindResult(color_name=u'bright turquoise', distance=3.730288645055483)
          >>> tint_registry.find_nearest("54e6e4", "en", filter_set=("white", "black"))
          FindResult(color_name=u'white', distance=25.709952192116894)

        """

        if system not in self._colors_by_system_hex:
            raise ValueError(
                "%r is not a registered color system. Try one of %r"
                % (system, self._colors_by_system_hex.keys())
            )
        hex_code = hex_code.lower().strip()

        # Try direct hit (fast path)
        if hex_code in self._colors_by_system_hex[system]:
            color_name = self._colors_by_system_hex[system][hex_code]
            if filter_set is None or color_name in filter_set:
                return FindResult(color_name, 0)

        # No direct hit, assemble list of lab_color/color_name pairs
        colors = self._colors_by_system_lab[system]
        if filter_set is not None:
            colors = (pair for pair in colors if pair[1] in set(filter_set))

        # find minimal distance
        lab_color = _hex_to_lab(hex_code)
        min_distance = sys.float_info.max
        min_color_name = None
        for current_lab_color, current_color_name in colors:
            distance = colormath.color_diff.delta_e_cie2000(lab_color, current_lab_color)
            if distance < min_distance:
                min_distance = distance
                min_color_name = current_color_name

        return FindResult(min_color_name, min_distance)