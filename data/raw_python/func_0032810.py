def format(self, format_str):
        """Returns a formatted version of format_str.
        The only named replacement fields supported by this method and
        their corresponding API calls are:

        * {num}           group_num
        * {name}          group_name
        * {symbol}        group_symbol
        * {variant}       group_variant
        * {current_data}  group_data
        * {nums}          groups_nums
        * {names}         groups_names
        * {symbols}       groups_symbols
        * {variants}      groups_variants
        * {all_data}      groups_data

        Passing other replacement fields will result in raising exceptions.

        :param format_str: a new style format string
        :rtype: str
        """
        return format_str.format(**{
            "num": self.group_num,
            "name": self.group_name,
            "symbol": self.group_symbol,
            "variant": self.group_variant,
            "current_data": self.group_data,
            "count": self.groups_count,
            "names": self.groups_names,
            "symbols": self.groups_symbols,
            "variants": self.groups_variants,
            "all_data": self.groups_data})