def _strip(text):
        """Strip articles/whitespace and remove case."""
        text = text.strip()
        text = text.replace('  ', ' ')  # remove duplicate spaces
        text = text.lower()
        for joiner in TextTitle.JOINERS:
            text = text.replace(joiner, 'and')
        for article in TextTitle.ARTICLES:
            if text.startswith(article + ' '):
                text = text[len(article) + 1:]
                break
        return text