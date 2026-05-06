def add_pseudo_fields(self):
        """Add 'pseudo' fields (e.g non-displayed fields) to the display."""
        fields = []
        if self.backlight_on != enums.BACKLIGHT_ON_NEVER:
            fields.append(
                display_fields.BacklightPseudoField(ref='0', backlight_rule=self.backlight_on)
            )

        fields.append(
            display_fields.PriorityPseudoField(
                ref='0',
                priority_playing=self.priority_playing,
                priority_not_playing=self.priority_not_playing,
            )
        )

        self.pattern.add_pseudo_fields(fields, self.screen)