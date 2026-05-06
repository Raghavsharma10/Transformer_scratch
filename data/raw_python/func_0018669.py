def _compare_dimensions_to_fields(self) -> bool:
        """ Compares the dimension field values to the value in regular fields."""
        for name, item in self._dimension_fields.items():
            if item.value != self._nested_items[name].value:
                return False
        return True