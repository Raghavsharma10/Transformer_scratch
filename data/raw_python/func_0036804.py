def title(self) -> str:
        """Episode title."""
        for title in self.titles:
            if title.lang == 'ja':
                return title.title
        # In case there's no Japanese title.
        return self.titles[0].title