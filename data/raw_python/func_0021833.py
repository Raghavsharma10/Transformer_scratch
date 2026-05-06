def show_distributions(self, show):
        """
        Show list of installed activated OR non-activated packages

        @param show: type of pkgs to show (all, active or nonactive)
        @type show: string

        @returns: None or 2 if error
        """
        show_metadata = self.options.metadata

        #Search for any plugins with active CLI options with add_column() method
        plugins = self.get_plugin("add_column")

        #Some locations show false positive for 'development' packages:
        ignores = ["/UNIONFS", "/KNOPPIX.IMG"]

        #Check if we're in a workingenv
        #See http://cheeseshop.python.org/pypi/workingenv.py
        workingenv = os.environ.get('WORKING_ENV')
        if workingenv:
            ignores.append(workingenv)

        dists = Distributions()
        results = None
        for (dist, active) in dists.get_distributions(show, self.project_name,
                self.version):
            metadata = get_metadata(dist)
            for prefix in ignores:
                if dist.location.startswith(prefix):
                    dist.location = dist.location.replace(prefix, "")
            #Case-insensitve search because of Windows
            if dist.location.lower().startswith(get_python_lib().lower()):
                develop = ""
            else:
                develop = dist.location
            if metadata:
                add_column_text = ""
                for my_plugin in plugins:
                    #See if package is 'owned' by a package manager such as
                    #portage, apt, rpm etc.
                    #add_column_text += my_plugin.add_column(filename) + " "
                    add_column_text += my_plugin.add_column(dist) + " "
                self.print_metadata(metadata, develop, active, add_column_text)
            else:
                print(str(dist) + " has no metadata")
            results = True
        if not results and self.project_name:
            if self.version:
                pkg_spec = "%s==%s" % (self.project_name, self.version)
            else:
                pkg_spec = "%s" % self.project_name
            if show == "all":
                self.logger.error("There are no versions of %s installed." \
                        % pkg_spec)
            else:
                self.logger.error("There are no %s versions of %s installed." \
                        % \
                        (show, pkg_spec))
            return 2
        elif show == "all" and results and self.options.fields:
            print("Versions with '*' are non-active.")
            print("Versions with '!' are deployed in development mode.")