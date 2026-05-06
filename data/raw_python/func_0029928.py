def as_dataset(self):
        """Convert this identity to the identity of the corresponding
        dataset."""

        on = self.on.dataset

        on.revision = self.on.revision

        name = Name(**self.name.dict)

        return Identity(name, on)