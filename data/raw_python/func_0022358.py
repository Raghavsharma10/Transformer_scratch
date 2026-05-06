def total_marks(self):
        """Compute the total mark for the assessment."""
        total = 0
        for answer in self.answers:
            for number, part in enumerate(answer):
                if number>0:
                    if part[2]>0:
                        total+=part[2]
        return total