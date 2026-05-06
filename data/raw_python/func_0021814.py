def get_distributions(self, show, pkg_name="", version=""):
        """
        Yield installed packages

        @param show: Type of package(s) to show; active, non-active or all
        @type show: string: "active", "non-active", "all"

        @param pkg_name: PyPI project name
        @type pkg_name: string

        @param version: project's PyPI version
        @type version: string

        @returns: yields tuples of distribution and True or False depending
                  on active state. e.g. (dist, True)

        """
        #pylint: disable-msg=W0612
        #'name' is a placeholder for the sorted list
        for name, dist in self.get_alpha(show, pkg_name, version):
            ver = dist.version
            for package in self.environment[dist.project_name]:
                if ver == package.version:
                    if show == "nonactive" and dist not in self.working_set:
                        yield (dist, self.query_activated(dist))
                    elif show == "active" and dist in self.working_set:
                        yield (dist, self.query_activated(dist))
                    elif show == "all":
                        yield (dist, self.query_activated(dist))