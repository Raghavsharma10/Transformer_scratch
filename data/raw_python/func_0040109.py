def convert_content(self) -> dict:
        """Convert content of source file into dict result."""
        source_content = self.load_content()

        converted = {}

        tagged, content = self._get_tags_and_content(source_content)
        if tagged:
            converted.update(self._parse_tags(tagged))

        if content:
            converted['content'] = content.strip()
        return converted