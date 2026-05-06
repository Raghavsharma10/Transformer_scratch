def select_release(self, highest_allowed_release):
        """
        Select the newest release that is not newer than the given release.

        :param highest_allowed_release: The identifier of the release that sets
                                        the upper bound for the selection (a
                                        string).
        :returns: The identifier of the selected release (a string).
        :raises: :exc:`~vcs_repo_mgr.exceptions.NoMatchingReleasesError`
                 when no matching releases are found.
        """
        matching_releases = []
        highest_allowed_key = natsort_key(highest_allowed_release)
        for release in self.ordered_releases:
            release_key = natsort_key(release.identifier)
            if release_key <= highest_allowed_key:
                matching_releases.append(release)
        if not matching_releases:
            msg = "No releases below or equal to %r found in repository!"
            raise NoMatchingReleasesError(msg % highest_allowed_release)
        return matching_releases[-1]