def load_zipfile(self, path):
        """
        import contents of a zipfile
        """
        # try to add as zipfile
        zin = zipfile.ZipFile(path)
        for zinfo in zin.infolist():
            name = zinfo.filename
            if name.endswith("/"):
                self.mkdir(name)
            else:
                content = zin.read(name)
                self.touch(name, content)