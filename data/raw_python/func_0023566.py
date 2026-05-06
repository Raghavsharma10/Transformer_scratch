def read(self, limit=-1):
        """Read content. See file.read"""
        remaining = self.len - self.parent_fd.tell() + self.offset

        if limit > remaining or limit == -1:
            limit = remaining

        return self.parent_fd.read(limit)