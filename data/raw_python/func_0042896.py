def pretty_print(self, printer: Optional[Printer] = None, min_width: int = 1, min_unit_width: int = 1):
        """
        Prints the file size (and it's unit), reserving places for longer sizes and units.
        For example:
            min_unit_width = 1:
                793 B
                100 KB
            min_unit_width = 2:
                793  B
                100 KB
            min_unit_width = 3:
                793   B
                100  KB
        """
        unit, unit_divider = self._unit_info()
        unit_color = self.SIZE_COLORS[unit]
        # Multiply and then divide by 100 in order to have only two decimal places.
        size_in_unit = (self.size * 100) / unit_divider / 100
        # Add spaces to align the units.
        unit = '{}{}'.format(' ' * (min_unit_width - len(unit)), unit)
        size_string = f'{size_in_unit:.1f}'
        total_len = len(size_string) + 1 + len(unit)
        if printer is None:
            printer = get_printer()
        spaces_count = min_width - total_len
        if spaces_count > 0:
            printer.write(' ' * spaces_count)
        printer.write(f'{size_string} {unit_color}{unit}')