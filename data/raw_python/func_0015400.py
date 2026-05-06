def find_vext_files(self):
        """
        :return:  Absolute paths to any provided vext files
        """
        packages = self.depends_on("vext")
        vext_files = []
        for location in [package.get("location") for package in packages]:
            if not location:
                continue
            vext_files.extend(glob(join(location, "*.vext")))
        return vext_files