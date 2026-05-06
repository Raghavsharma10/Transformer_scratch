def filenames(self):
        """Returns the filenames that this par2 file repairs."""
        return [p.name for p in self.packets if isinstance(p, FileDescriptionPacket)]