def _split_docstring(self, docstring):
        """Separate a docstring into the synopsis (first line) and body."""

        lines = docstring.strip().splitlines()

        synopsis = lines[0].strip()
        body = textwrap.dedent('\n'.join(lines[2:]))

        # Remove RST preformatted text markers.
        body = body.replace('\n::\n', '')
        body = body.replace('::\n', ':')

        return (synopsis, body)