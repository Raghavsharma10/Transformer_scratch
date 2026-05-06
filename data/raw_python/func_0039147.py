def get_dependants(cls, dist):
        """Yield dependant user packages for a given package name."""
        for package in cls.installed_distributions:
            for requirement_package in package.requires():
                requirement_name = requirement_package.project_name
                # perform case-insensitive matching
                if requirement_name.lower() == dist.lower():
                    yield package