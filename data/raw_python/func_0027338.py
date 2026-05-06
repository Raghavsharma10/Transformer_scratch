def export_to_file(self, filename):
        """Export this instrument's settings to a file.

        :param filename: the name of the file
        """
        instr_json = self.export_struct()

        with open(filename, 'w') as fp:
            json.dump(instr_json, fp, indent=2)