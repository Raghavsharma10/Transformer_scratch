def remove_graphic(self, graphic: Graphics.Graphic, *, safe: bool=False) -> typing.Optional[typing.Sequence]:
        """Remove a graphic, but do it through the container, so dependencies can be tracked."""
        return self.remove_model_item(self, "graphics", graphic, safe=safe)