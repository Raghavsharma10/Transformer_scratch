def delete(self):
        """Remove the statement from a ProcmailRC structure, raise a
        RuntimeError if the statement is not inside a ProcmailRC structure
        return the parent id"""
        if self.parent is None:
            raise RuntimeError(
                "Current statement has no parent, so it cannot "
                + "be deleted form a procmailrc structure"
            )
        elif self.id is None:
            raise RuntimeError("id not set but have a parent, this should not be happening")
        else:
            parent_id = self.parent.id
            index = int(self.id.split('.')[-1])
            self.parent.pop(index)
            self.parent = None
            self.id = None
            return parent_id