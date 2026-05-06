def generate(self):
    """Generate a list of strings representing the table in RST format."""
    header = ' '.join('=' * self.width[i] for i in range(self.w))
    lines = [
        ' '.join(row[i].ljust(self.width[i]) for i in range(self.w))
        for row in self.rows]
    return [header] + lines + [header]