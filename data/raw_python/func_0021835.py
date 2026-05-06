def show_deps(self):
        """
        Show dependencies for package(s)

        @returns: 0 - sucess  1 - No dependency info supplied
        """

        pkgs = pkg_resources.Environment()

        for pkg in pkgs[self.project_name]:
            if not self.version:
                print(pkg.project_name, pkg.version)

            i = len(pkg._dep_map.values()[0])
            if i:
                while i:
                    if not self.version or self.version and \
                            pkg.version == self.version:
                        if self.version and i == len(pkg._dep_map.values()[0]):
                            print(pkg.project_name, pkg.version)
                        print("  " + str(pkg._dep_map.values()[0][i - 1]))
                    i -= 1
            else:
                self.logger.info(\
                    "No dependency information was supplied with the package.")
                return 1
        return 0