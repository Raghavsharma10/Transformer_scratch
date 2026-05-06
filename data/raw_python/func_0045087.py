def _print_fields(self, fields):
        """Print the fields, padding the names as necessary to align them."""
        # Prepare a formatting string that aligns the names and types based on the longest ones
        longest_name = max(fields, key=lambda f: len(f[1]))[1]
        longest_type = max(fields, key=lambda f: len(f[2]))[2]
        field_format = '%s%-{}s %-{}s %s'.format(
            len(longest_name) + self._padding_after_name,
            len(longest_type) + self._padding_after_type)
        for field in fields:
            self._print(field_format % field)