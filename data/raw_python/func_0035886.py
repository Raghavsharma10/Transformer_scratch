def _get_str(self, f, off):
        """
        Convenience function to quickly pull out strings.
        """
        f.seek(off)
        return f.read(2 * struct.unpack('>B', f.read(1))[0]).decode('utf-16')