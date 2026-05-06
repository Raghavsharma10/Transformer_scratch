def _parse_result(self, ok, match, fh=None):
        """Parse a matching result line into a result instance."""
        peek_match = None
        try:
            if fh is not None and self._try_peeking:
                peek_match = self.yaml_block_start.match(fh.peek())
        except StopIteration:
            pass
        if peek_match is None:
            return Result(
                ok,
                number=match.group("number"),
                description=match.group("description").strip(),
                directive=Directive(match.group("directive")),
            )
        indent = peek_match.group("indent")
        concat_yaml = self._extract_yaml_block(indent, fh)
        return Result(
            ok,
            number=match.group("number"),
            description=match.group("description").strip(),
            directive=Directive(match.group("directive")),
            raw_yaml_block=concat_yaml,
        )