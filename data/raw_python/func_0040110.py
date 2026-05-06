def _get_tags_and_content(self, content: str) -> typing.Tuple[str, str]:
        """Splits content into two string - tags part and another content."""
        content_lines = content.split('\n')
        tag_lines = []

        if content_lines[0] != '---':
            return '', content

        content_lines.pop(0)
        for line in content_lines:  # type: str
            if line in ('---', '...'):
                content_starts_at = content_lines.index(line) + 1
                content_lines = content_lines[content_starts_at:]
                break

            tag_lines.append(line)

        return '\n'.join(tag_lines), '\n'.join(content_lines)