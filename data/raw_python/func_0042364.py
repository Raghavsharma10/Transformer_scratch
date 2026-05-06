def write_title(self, title: str, title_color: str = YELLOW, hyphen_line_color: str = WHITE):
        """
        Prints title with hyphen line underneath it.

        :param title: The title to print.
        :param title_color: The title text color (default is yellow).
        :param hyphen_line_color: The hyphen line color (default is white).
        """
        self.write_line(title_color + title)
        self.write_line(hyphen_line_color + '=' * (len(title) + 3))