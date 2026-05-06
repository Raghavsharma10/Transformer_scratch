def output(self, full_text, short_text):
        """
        Output all of the options and data for a segment.

        full_text: A string representing the data that should be output to i3bar.
        short_text: A more concise version of full_text, in case there is minimal
        room on the i3bar.
        """
        full_text = full_text.replace('\n', '')
        short_text = short_text.replace('\n', '')
        self.output_options.update({'full_text': full_text, 'short_text': short_text})
        self.output_options = {k: v for k, v in self.output_options.items() if v}
        return self.output_options