def validate_version(self, prefix='v'):
        """Validate version by checking if it is a valid semantic version
        and its value is higher than latest github tag
        """
        version = self.software_version()
        repo = self.github_repo()
        repo.releases.validate_tag(version, prefix)
        return version