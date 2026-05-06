def uuid(self):
        """
        Returns the volume group uuid.
        """
        self.open()
        uuid = lvm_vg_get_uuid(self.handle)
        self.close()
        return uuid