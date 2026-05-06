def _info_long(self) -> Optional[str]:
        """Extract journey information."""
        try:
            return str(
                html.unescape(self.journey.InfoTextList.InfoText.get("textL")).replace(
                    "<br />", "\n"
                )
            )
        except AttributeError:
            return None