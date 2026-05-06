def _validate_node_name(self, var_value):
        """Validate NodeName pseudo-type."""
        # pylint: disable=R0201
        var_values = var_value if isinstance(var_value, list) else [var_value]
        for item in var_values:
            if (not isinstance(item, str)) or (
                isinstance(item, str)
                and (
                    (" " in item)
                    or any(
                        [
                            element.strip() == ""
                            for element in item.strip().split(self._node_separator)
                        ]
                    )
                )
            ):
                return True
        return False