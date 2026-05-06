def _remove_empty_output(self):
        """
        If plugins haven't been initialised and therefore not sending output or
        their output is None, there is no reason to take up extra room on the
        bar.
        """
        output = []
        for key in self.output_dict:
            if self.output_dict[key] is not None and 'full_text' in self.output_dict[key]:
                output.append(self.output_dict[key])
        return output