def to_tex(self, text_size='large', table_width=5, clear_pages = False):
        """
        Write the program information to a .tex file, which can be
        rendered to .pdf running pdflatex. The program can then be
        printed and brought to the gym.

        Parameters
        ----------
        text_size
            The tex text size, e.g. '\small', 'normalsize', 'large', 'Large'
            or 'LARGE'.

        table_width
            The table with of the .tex code.

        Returns
        -------
        string
            Program as tex.
        """

        # If rendered, find the length of the longest '6 x 75kg'-type string
        max_ex_scheme = 0
        if self._rendered:
            for (week, day, dynamic_ex) in self._yield_week_day_dynamic():
                lengths = [len(s) for s in
                           self._rendered[week][day][dynamic_ex]['strings']]
                max_ex_scheme = max(max_ex_scheme, max(lengths))


        env = self.jinja2_environment
        template = env.get_template(self.TEMPLATE_NAMES['tex'])

        return template.render(program=self, text_size=text_size,
                               table_width=table_width, clear_pages = clear_pages)