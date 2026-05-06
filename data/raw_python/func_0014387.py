def to_json(self):
        """
        Returns the JSON representation of the snapshot.
        """

        result = super(Snapshot, self).to_json()
        result.update({
            'snapshot': self.snapshot.to_json(),
        })
        return result