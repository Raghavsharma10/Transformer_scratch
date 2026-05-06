def set_cache_complex_value(self, name, value):
        """Set a variable in the local complex state dictionary.

        This does not change the physical device. Useful if you want the
        device state to refect a new value which has not yet updated from
        Vera.
        """
        for item in self.json_state.get('states'):
            if item.get('variable') == name:
                item['value'] = str(value)