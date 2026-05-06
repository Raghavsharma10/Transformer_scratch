def write_aligned(self, key: str, value: str, not_important_keys: Optional[List[str]] = None,
                      is_list: bool = False, align_size: Optional[int] = None, key_color: str = PURPLE,
                      value_color: str = GREEN, dark_key_color: str = DARK_PURPLE, dark_value_color: str = DARK_GREEN,
                      separator: str = SEPARATOR):
        """
        Prints keys and values aligned to align_size.

        :param key: The name of the property to print.
        :param value: The value of the property to print.
        :param not_important_keys: Properties that will be printed in a darker color.
        :param is_list: True if the value is a list of items.
        :param align_size: The alignment size to use.
        :param key_color: The key text color (default is purple).
        :param value_color: The value text color (default is green).
        :param dark_key_color: The key text color for unimportant keys (default is dark purple).
        :param dark_value_color: The values text color for unimportant values (default is dark green).
        :param separator: The separator to use (default is ':').
        """
        align_size = align_size or min(32, get_console_width() // 2)
        not_important_keys = not_important_keys or []
        if value is None:
            return
        if isinstance(value, bool):
            value = str(value)
        if key in not_important_keys:
            key_color = dark_key_color
            value_color = dark_value_color

        self.write(key_color + key + separator)
        self.write(' ' * (align_size - len(key) - 1))
        with self.group(indent=align_size):
            if is_list and len(value) > 0:
                self.write_line(value_color + value[0])
                if len(value) > 1:
                    for v in value[1:]:
                        self.write_line(value_color + v)
            elif not is_list:
                self.write_line(value_color + str(value))