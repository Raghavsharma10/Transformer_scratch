def compare_config(self):
        """Compare candidate config with running."""
        diff = self.device.cu.diff()

        if diff is None:
            return ''
        else:
            return diff.strip()