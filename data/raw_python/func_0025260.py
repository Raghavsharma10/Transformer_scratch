def start(self, value: typing.Union[float, NormPointType]) -> None:
        """Set the end property in relative coordinates.

        End may be a float when graphic is an Interval or a tuple (y, x) when graphic is a Line."""
        self.set_property("start", value)