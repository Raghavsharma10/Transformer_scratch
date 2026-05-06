def _get_next_version(self, revisions):
        """
        Calculates new version number based on existing numeric ones.
        """
        versions = [0]
        for v in revisions:
            if v.isdigit():
                versions.append(int(v))
        return six.text_type(sorted(versions)[-1] + 1)