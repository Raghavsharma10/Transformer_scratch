def releasers(cls):
        """
        Returns all of the supported releasers.
        """

        return [
            HookReleaser,
            VersionFileReleaser,
            PythonReleaser,
            CocoaPodsReleaser,
            NPMReleaser,
            CReleaser,
            ChangelogReleaser,
            GitHubReleaser,
            GitReleaser,
        ]