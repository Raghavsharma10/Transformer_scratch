def get_tagged_version(self):
		"""
		Get the version of the local working set as a StrictVersion or
		None if no viable tag exists. If the local working set is itself
		the tagged commit and the tip and there are no local
		modifications, use the tag on the parent changeset.
		"""
		tags = list(self.get_tags())
		if 'tip' in tags and not self.is_modified():
			tags = self.get_parent_tags('tip')
		versions = self.__versions_from_tags(tags)
		return self.__best_version(versions)