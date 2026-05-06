def cli_program_names(self):
        r"""Developer script program names.
        """
        program_names = {}
        for cli_class in self.cli_classes:
            instance = cli_class()
            program_names[instance.program_name] = cli_class
        return program_names