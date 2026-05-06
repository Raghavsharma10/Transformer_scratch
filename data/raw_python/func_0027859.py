def is_clustered(self):
        """
        Returns True if the VG is clustered, False otherwise.
        """
        self.open()
        clust = lvm_vg_is_clustered(self.handle)
        self.close()
        return bool(clust)