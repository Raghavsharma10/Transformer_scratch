def parse_pkg_ver(self, want_installed):
        """
        Return tuple with project_name and version from CLI args
        If the user gave the wrong case for the project name, this corrects it

        @param want_installed: whether package we want is installed or not
        @type want_installed: boolean

        @returns: tuple(project_name, version, all_versions)

        """
        all_versions = []

        arg_str = ("").join(self.pkg_spec)
        if "==" not in arg_str:
            #No version specified
            project_name = arg_str
            version = None
        else:
            (project_name, version) = arg_str.split("==")
            project_name = project_name.strip()
            version = version.strip()
        #Find proper case for package name
        if want_installed:
            dists = Distributions()
            project_name = dists.case_sensitive_name(project_name)
        else:
            (project_name, all_versions) = \
                    self.pypi.query_versions_pypi(project_name)

            if not len(all_versions):
                msg = "I'm afraid we have no '%s' at " % project_name
                msg += "The Cheese Shop. A little Red Leicester, perhaps?"
                self.logger.error(msg)
                sys.exit(2)
        return (project_name, version, all_versions)