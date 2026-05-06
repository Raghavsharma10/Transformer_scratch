def to_txt(self, verbose=False):
        """Write the program information to text,
        which can be printed in a terminal.

        Parameters
        ----------
        verbose
            If True, more information is shown.

        Returns
        -------
        string
            Program as text.
        """
        # Get information related to formatting
        exercises = list(self._yield_exercises())
        max_ex_name = 0
        if len(exercises) != 0:
            max_ex_name = max(len(ex.name) for ex in exercises)

        # If rendered, find the length of the longest '6 x 75kg'-type string
        max_ex_scheme = 0
        if self._rendered:
            for (week, day, dynamic_ex) in self._yield_week_day_dynamic():
                lengths = [len(s) for s in
                           self._rendered[week][day][dynamic_ex]['strings']]
                max_ex_scheme = max(max_ex_scheme, max(lengths))


        env = self.jinja2_environment
        template = env.get_template(self.TEMPLATE_NAMES['txt'])
        return template.render(program=self, max_ex_name=max_ex_name,
                               max_ex_scheme=max_ex_scheme, verbose=verbose)