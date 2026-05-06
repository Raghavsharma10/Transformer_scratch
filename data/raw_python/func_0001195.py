def split_flanks(self, _, result):
        """Return `result` without flanking whitespace.
        """
        if not result.strip():
            self.left, self.right = "", ""
            return result

        match = self.flank_re.match(result)
        assert match, "This regexp should always match"
        self.left, self.right = match.group(1), match.group(3)
        return match.group(2)