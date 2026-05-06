def get_info(self):
        """
        Parses the output of a "show info" HAProxy command and returns a
        simple dictionary of the results.
        """
        info_response = self.send_command("show info")

        if not info_response:
            return {}

        def convert_camel_case(string):
            return all_cap_re.sub(
                r'\1_\2',
                first_cap_re.sub(r'\1_\2', string)
            ).lower()

        return dict(
            (convert_camel_case(label), value)
            for label, value in [
                line.split(": ")
                for line in info_response.split("\n")
            ]
        )