def _delay(self) -> int:
        """Extract departure delay."""
        try:
            return int(self.journey.MainStop.BasicStop.Dep.Delay.text)
        except AttributeError:
            return 0