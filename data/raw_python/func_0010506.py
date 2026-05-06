def _process_templatedata(self, node, **_):
        """
        Processes a `TemplateData` node, this is just a bit of as-is text
        to be written to the output.
        """

        # escape double quotes
        value = re.sub('"', r'\\"', node.data)

        # escape new lines
        value = re.sub('\n', r'\\n', value)

        # append value to the result
        self.output.write('__result += "' + value + '";')