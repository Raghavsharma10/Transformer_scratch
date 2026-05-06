def render(self):
        """
        print provided table

        :return: None
        """
        print(self.format_str.format(**self.header), file=sys.stderr)
        print(self.header_format_str.format(**self.header_data), file=sys.stderr)
        for row in self.data:
            print(self.format_str.format(**row))