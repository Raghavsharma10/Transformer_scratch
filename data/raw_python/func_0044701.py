def _dedent(text):
        """Remove common indentation from each line in a text block.

        When text block is a single line, return text block. Otherwise
        determine common indentation from last line, strip common
        indentation from each line, and return text block consisting of
        inner lines (don't include first and last lines since they either
        empty or contain whitespace and are present in baselined
        string to make them pretty and delineate the common indentation).

        :param str text: text block
        :returns: text block with common indentation removed
        :rtype: str
        :raises ValueError: when text block violates whitespace rules

        """
        lines = text.split('\n')

        if len(lines) == 1:
            indent = 0

        elif lines[0].strip():
            raise ValueError('when multiple lines, first line must be blank')

        elif lines[-1].strip():
            raise ValueError('last line must only contain indent whitespace')

        else:
            indent = len(lines[-1])

            if any(line[:indent].strip() for line in lines):
                raise ValueError(
                    'indents must equal or exceed indent in last line')

            lines = [line[indent:] for line in lines][1:-1]

        return indent, '\n'.join(lines)