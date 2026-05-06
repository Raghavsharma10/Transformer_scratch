def skip(self):
        """Skip this py-pdb command to avoid attaching within the same loop."""

        line = self.line
        self.line = ''
        # 'line' is the statement line of the previous py-pdb command.
        if line in self.lines:
            if not self.skipping:
                self.skipping = True
                printflush('Skipping lines', end='')
            printflush('.', end='')
            return True
        elif line:
            self.lines.append(line)
            if len(self.lines) > 30:
                self.lines.popleft()

        return False