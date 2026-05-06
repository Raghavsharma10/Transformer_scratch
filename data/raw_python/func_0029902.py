def promote(self, name):
        """Promote to a PartitionName by combining with a bundle Name."""
        return PartitionName(**dict(list(name.dict.items()) + list(self.dict.items())))