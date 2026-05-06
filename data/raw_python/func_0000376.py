def run(self, input_func=_stdin_):
        """Run the sections."""
        # reset question count
        self.qcount = 1
        for section_name in self.survey:
            self.run_section(section_name, input_func)