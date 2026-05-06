def find_branches(self):
        """Find information about the branches in the repository."""
        for prefix, name, revision_id in self.find_branches_raw():
            yield Revision(
                branch=name,
                repository=self,
                revision_id=revision_id,
            )