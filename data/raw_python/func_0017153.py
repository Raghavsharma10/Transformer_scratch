def assign(self, login):
        """Assigns user ``login`` to this issue. This is a short cut for
        ``issue.edit``.

        :param str login: username of the person to assign this issue to
        :returns: bool
        """
        if not login:
            return False
        number = self.milestone.number if self.milestone else None
        labels = [str(l) for l in self.labels]
        return self.edit(self.title, self.body, login, self.state, number,
                         labels)