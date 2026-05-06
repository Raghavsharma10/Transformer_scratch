def _platform(self) -> Optional[str]:
        """Extract platform."""
        try:
            return str(self.journey.MainStop.BasicStop.Dep.Platform.text)
        except AttributeError:
            return None