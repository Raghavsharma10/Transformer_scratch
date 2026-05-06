def main(self):
        """
        Generates an output string by replacing the keywords in the format
        string with the corresponding values from a submission dictionary.
        """
        self.manage_submissions()
        out_string = self.options['format']

        # Pop until we get something which len(title) <= max-chars
        length = float('inf')
        while length > self.options['max_chars']:
            self.selected_submission = self.submissions.pop()
            length = len(self.selected_submission['title'])

        for k, v in self.selected_submission.items():
            out_string = out_string.replace(k, self.h.unescape(str(v)))
        return self.output(out_string, out_string)