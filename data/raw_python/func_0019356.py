def read(self) -> None:
        """Call method |NetCDFFile.read| of all handled |NetCDFFile| objects.
        """
        for folder in self.folders.values():
            for file_ in folder.values():
                file_.read()