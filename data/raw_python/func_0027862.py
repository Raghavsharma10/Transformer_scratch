def sequence(self):
        """
        Returns the volume group sequence number. This number increases
        everytime the volume group is modified.
        """
        self.open()
        seq = lvm_vg_get_seqno(self.handle)
        self.close()
        return seq