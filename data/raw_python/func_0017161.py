def reopen(self):
        """Re-open a closed issue.

        :returns: bool
        """
        assignee = self.assignee.login if self.assignee else ''
        number = self.milestone.number if self.milestone else None
        labels = [str(l) for l in self.labels]
        return self.edit(self.title, self.body, assignee, 'open',
                         number, labels)