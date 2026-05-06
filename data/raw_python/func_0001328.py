def format_config_read_queue(self,
                                 use_color: bool = False,
                                 max_col_width: int = 50) -> str:
        """
        Prepares a string with pretty printed config read queue.

        :param use_color: use terminal colors
        :param max_col_width: limit column width, ``50`` by default
        :return:
        """
        try:
            from terminaltables import SingleTable
        except ImportError:
            import warnings
            warnings.warn('Cannot display config read queue. Install terminaltables first.')
            return ''

        col_names_order = ['path', 'value', 'type', 'parser']
        pretty_bundles = [[self._colorize(name, name.capitalize(), use_color=use_color)
                           for name in col_names_order]]

        for config_read_item in self.config_read_queue:
            pretty_attrs = [
                config_read_item.variable_path,
                config_read_item.value,
                config_read_item.type,
                config_read_item.parser_name
            ]
            pretty_attrs = [self._pformat(pa, max_col_width) for pa in pretty_attrs]

            if config_read_item.is_default:
                pretty_attrs[0] = '*' + pretty_attrs[0]

            if use_color:
                pretty_attrs = [self._colorize(column_name, pretty_attr, use_color=use_color)
                                for column_name, pretty_attr in zip(col_names_order, pretty_attrs)]
            pretty_bundles.append(pretty_attrs)

        table = SingleTable(pretty_bundles)
        table.title = self._colorize('title', 'CONFIG READ QUEUE', use_color=use_color)
        table.justify_columns[0] = 'right'
        # table.inner_row_border = True
        return str(table.table)