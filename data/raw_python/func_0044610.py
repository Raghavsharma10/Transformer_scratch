def _build_tags(self, tag_names: List[str]) -> dict:
        """Build a list of tag objects."""
        tags = {}
        for tag_name in tag_names:
            tag_obj = self.tag(tag_name)
            if tag_obj is None:
                LOG.debug(f"create new tag: {tag_name}")
                tag_obj = self.new_tag(tag_name)
            tags[tag_name] = tag_obj
        return tags