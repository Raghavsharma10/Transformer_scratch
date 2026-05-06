def _parse_by_pattern(self, lines, pattern):
        """Match pattern line by line and return Results.

        Use ``_create_output_from_match`` to convert pattern match groups to
        Result instances.

        Args:
            lines (iterable): Output lines to be parsed.
            pattern: Compiled pattern to match against lines.
            result_fn (function): Receive results of one match and return a
                Result.

        Return:
            generator: Result instances.
        """
        for line in lines:
            match = pattern.match(line)
            if match:
                params = match.groupdict()
                if not params:
                    params = match.groups()
                yield self._create_output_from_match(params)