def save(self, path, compressed=True, exist_ok=False):
        """
        Save the GADDAG to file.

        Args:
            path: path to save the GADDAG to.
            compressed: compress the saved GADDAG using gzip.
            exist_ok: overwrite existing file at `path`.
        """
        path = os.path.expandvars(os.path.expanduser(path))
        if os.path.isfile(path) and not exist_ok:
            raise OSError(17, os.strerror(17), path)

        if os.path.isdir(path):
            path = os.path.join(path, "out.gdg")

        if compressed:
            bytes_written = cgaddag.gdg_save_compressed(self.gdg, path.encode("ascii"))
        else:
            bytes_written = cgaddag.gdg_save(self.gdg, path.encode("ascii"))

        if bytes_written == -1:
            errno = ctypes.c_int.in_dll(ctypes.pythonapi, "errno").value
            raise OSError(errno, os.strerror(errno), path)

        return bytes_written