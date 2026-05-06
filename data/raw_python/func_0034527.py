def conflicts(self, ext):
        """
        Check if the extension conflicts with an already accepted extension.
        This may be the case when the two extensions use the same reserved
        bits, or have the same name (when the same extension is negotiated
        multiple times with different parameters).
        """
        return ext.rsv1 and self.rsv1 \
            or ext.rsv2 and self.rsv2 \
            or ext.rsv3 and self.rsv3 \
            or set(ext.names) & set(self.names) \
            or set(ext.opcodes) & set(self.opcodes)