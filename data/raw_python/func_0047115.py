def to_html(self, table_width=5):
        """Write the program information to HTML code, which can be saved,
        printed and brought to the gym.

        Parameters
        ----------
        table_width
            The table with of the HTML code.

        Returns
        -------
        string
            HTML code.
        """

        env = self.jinja2_environment
        template = env.get_template(self.TEMPLATE_NAMES['html'])
        return template.render(program=self, table_width=table_width)