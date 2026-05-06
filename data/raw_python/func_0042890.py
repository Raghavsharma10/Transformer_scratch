def dealwith(self, readline, **kwargs):
        """
            Replace the contents of spec file with the translated version
            readline should be a callable object
            , which provides the same interface as the readline() method of built-in file objects
        """
        data = []
        try:
            # We pass in the data variable as an argument so that we
            # get partial output even in the case of an exception.
            self.tokeniser.translate(readline, data, **kwargs)
        except Exception as e:
            # Comment out partial output so that it doesn't result in
            # a syntax error when received by the interpreter.
            lines = []
            for line in untokenize(data).split('\n'):
                lines.append("# {0}".format(line))

            # Create exception to put into code to announce error
            exception = 'raise Exception("""--- internal spec codec error --- {0}""")'.format(e)

            # Need to make sure the exception doesn't add a new line and put out line numberes
            if len(lines) == 1:
                data = "{0}{1}".format(exception, lines[0])
            else:
                lines.append(exception)
                first_line = lines.pop()
                lines[0] = "{0} {1}".format(first_line, lines[0])
                data = '\n'.join(lines)
        else:
            # At this point, data is a list of tokens
            data = untokenize(data)

        if six.PY2 and type(data) is not unicode:
            data = unicode(data)

        return data