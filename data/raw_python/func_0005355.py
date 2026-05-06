def convert_to_argument(self):
        '''
            Convert the Argument object to a tuple use in :meth:`~argparse.ArgumentParser.add_argument` calls on the parser
        '''

        field_list = [
            "action", "nargs", "const", "default", "type",
            "choices", "required", "help", "metavar", "dest"
        ]

        return (
            self.name,
            {
                field: getattr(self, field) for field in field_list if getattr(self, field) is not None
            }
        )