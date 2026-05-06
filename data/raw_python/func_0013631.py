def parseStep(self, line):
        """
        Parse the line describing the mode.

        One of:
        variableStep chrom=<reference> [span=<window_size>]
        fixedStep chrom=<reference> start=<position> step=<step_interval>
                  [span=<window_size>]

        Span is optional, defaulting to 1. It indicates that each value
        applies to region, starting at the given position and extending
        <span> positions.
        """
        fields = dict([field.split('=') for field in line.split()[1:]])

        if 'chrom' in fields:
            self._reference = fields['chrom']
        else:
            raise ValueError("Missing chrom field in %s" % line.strip())

        if line.startswith("fixedStep"):
            if 'start' in fields:
                self._start = int(fields['start']) - 1  # to 0-based
            else:
                raise ValueError("Missing start field in %s" % line.strip())

        if 'span' in fields:
            self._span = int(fields['span'])
        if 'step' in fields:
            self._step = int(fields['step'])