def dump(self):
        """Return the object itself."""

        return {
            'title': self.title,
            'issue_id': self.issue_id,
            'reporter': self.reporter,
            'assignee': self.assignee,
            'status': self.status,
            'product': self.product,
            'component': self.component,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'closed_at': self.closed_at,
            'status_code': self.status_code
        }