def variables(self):
        """
        Display a list of templatable variables present in the file.

        Templating is accomplished by creating a bracketed object in the same
        way that Python performs `string formatting`_. The editor is able to
        replace the placeholder value of the template. Integer templates are
        positional arguments.

        .. _string formatting: https://docs.python.org/3.6/library/string.html
        """
        string = str(self)
        constants = [match[1:-1] for match in re.findall('{{[A-z0-9]}}', string)]
        variables = re.findall('{[A-z0-9]*}', string)
        return sorted(set(variables).difference(constants))