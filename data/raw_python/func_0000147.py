def validate_tag(self, tag_name, prefix=None):
        """Validate ``tag_name`` with the latest tag from github

        If ``tag_name`` is a valid candidate, return the latest tag from github
        """
        new_version = semantic_version(tag_name)
        current = self.latest()
        if current:
            tag_name = current['tag_name']
            if prefix:
                tag_name = tag_name[len(prefix):]
            tag_name = semantic_version(tag_name)
            if tag_name >= new_version:
                what = 'equal to' if tag_name == new_version else 'older than'
                raise GithubException(
                    'Your local version "%s" is %s '
                    'the current github version "%s".\n'
                    'Bump the local version to '
                    'continue.' %
                    (
                        str(new_version),
                        what,
                        str(tag_name)
                    )
                )
        return current