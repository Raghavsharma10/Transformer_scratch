def print_config_read_queue(
            self,
            use_color: bool = False,
            max_col_width: int = 50):
        """
        Prints all read (in call order) options.

        :param max_col_width: limit column width, ``50`` by default
        :param use_color: use terminal colors
        :return: nothing
        """
        wf(self.format_config_read_queue(use_color=use_color, max_col_width=max_col_width))
        wf('\n')