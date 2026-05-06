def pick(self, filenames: Iterable[str]) -> str:
        """Pick one filename based on priority rules."""
        filenames = sorted(filenames, reverse=True)  # e.g., v2 before v1
        for priority in sorted(self.rules.keys(), reverse=True):
            patterns = self.rules[priority]
            for pattern in patterns:
                for filename in filenames:
                    if pattern.search(filename):
                        return filename
        return filenames[0]