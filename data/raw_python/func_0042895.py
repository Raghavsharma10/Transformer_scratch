def _unit_info(self) -> Tuple[str, int]:
        """
        Returns both the best unit to measure the size, and its power.

        :return: A tuple containing the unit and its power.
        """
        abs_bytes = abs(self.size)
        if abs_bytes < 1024:
            unit = 'B'
            unit_divider = 1
        elif abs_bytes < (1024 ** 2):
            unit = 'KB'
            unit_divider = 1024
        elif abs_bytes < (1024 ** 3):
            unit = 'MB'
            unit_divider = (1024 ** 2)
        elif abs_bytes < (1024 ** 4):
            unit = 'GB'
            unit_divider = (1024 ** 3)
        else:
            unit = 'TB'
            unit_divider = (1024 ** 4)

        return unit, unit_divider