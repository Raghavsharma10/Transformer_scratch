def changelog(self, api_version, doc):
        """Add a changelog entry for this api."""
        doc = textwrap.dedent(doc).strip()
        self._changelog[api_version] = doc
        self._changelog_locations[api_version] = get_callsite_location()