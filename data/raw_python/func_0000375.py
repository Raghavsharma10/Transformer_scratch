def run_section(self, name, input_func=_stdin_):
        """Run the given section."""
        print('\nStuff %s by the license:\n' % name)
        section = self.survey[name]
        for question in section:
            self.run_question(question, input_func)