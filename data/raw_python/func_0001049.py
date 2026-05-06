def _info(self) -> Optional[str]:
        """Extract journey information."""
        try:
            return str(html.unescape(self.journey.InfoTextList.InfoText.get("text")))
        except AttributeError:
            return None