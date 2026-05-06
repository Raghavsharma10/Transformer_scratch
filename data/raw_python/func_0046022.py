def check_version_consistency(self):
        """
        Determine if any releasers have inconsistent versions
        """

        version = None
        releaser_name = None

        for releaser in self.releasers:
            try:
                next_version = releaser.determine_current_version()
            except NotImplementedError:
                continue

            if next_version and version and version != next_version:
                raise Exception('Inconsistent versions, {} is at {} but {} is at {}.'.format(
                                releaser_name, version, releaser.name, next_version))

            version = next_version
            releaser_name = releaser.name