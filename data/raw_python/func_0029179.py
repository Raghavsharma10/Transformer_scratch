def parse_lines(self, code):
        """
        Return a list of the parsed code

        For each line, return a three-tuple containing:
        1. The label
        2. The instruction
        3. Any arguments or parameters

        An element in the tuple may be None or '' if it did not find anything
        :param code: The code to parse
        :return: A list of tuples in the form of (label, instruction, parameters)
        """
        remove_comments = re.compile(r'^([^;@\n]*);?.*$', re.MULTILINE)
        code = '\n'.join(remove_comments.findall(code))  # TODO can probably do this better
        # TODO labels with spaces between pipes is allowed `|label with space| INST OPER`
        parser = re.compile(r'^(\S*)?[\s]*(\S*)([^\n]*)$', re.MULTILINE)
        res = parser.findall(code)
        # Make all parsing of labels and instructions adhere to all uppercase
        res = [(label.upper(), instruction.upper(), parameters.strip()) for (label, instruction, parameters) in res]
        return res