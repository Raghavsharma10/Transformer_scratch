def visible_area(self):
        """
        Calculated like in the official client.
        Returns (top_left, bottom_right).
        """
        # looks like zeach has a nice big screen
        half_viewport = Vec(1920, 1080) / 2 / self.scale
        top_left = self.world.center - half_viewport
        bottom_right = self.world.center + half_viewport
        return top_left, bottom_right