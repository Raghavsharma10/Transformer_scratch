def get_alpha(self, show, pkg_name="", version=""):
        """
        Return list of alphabetized packages

        @param pkg_name: PyPI project name
        @type pkg_name: string

        @param version: project's PyPI version
        @type version: string

        @returns: Alphabetized list of tuples. Each tuple contains
                  a string and a pkg_resources Distribution object.
                  The string is the project name + version.

        """
        alpha_list = []
        for dist in self.get_packages(show):
            if pkg_name and dist.project_name != pkg_name:
                #Only checking for a single package name
                pass
            elif version and dist.version != version:
                #Only checking for a single version of a package
                pass
            else:
                alpha_list.append((dist.project_name + dist.version, dist))
        alpha_list.sort()
        return alpha_list