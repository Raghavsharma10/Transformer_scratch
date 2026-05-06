def _parse_tagfile(self):
        """Parse the tagfile and yield tuples of tag_name, list of rule ASTs."""
        rules = None
        tag = None
        for line in self.original:
            match = self.TAG_DECL_LINE.match(line)
            if match:
                if tag and rules:
                    yield tag, rules
                rules = []
                tag = match.group(1)
                continue

            match = self.TAG_RULE_LINE.match(line)
            if match:
                source = match.group(1)
                rules.append(self._parse_query(source))