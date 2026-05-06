def get_packages(self, show):
        """
        Return list of Distributions filtered by active status or all

        @param show: Type of package(s) to show; active, non-active or all
        @type show: string: "active", "non-active", "all"

        @returns: list of pkg_resources Distribution objects
        """


        if show == 'nonactive' or show == "all":
            all_packages = []
            for package in self.environment:
                #There may be multiple versions of same packages
                for i in range(len(self.environment[package])):
                    if self.environment[package][i]:
                        all_packages.append(self.environment[package][i])
            return all_packages
        else:
            # Only activated packages
            return self.working_set