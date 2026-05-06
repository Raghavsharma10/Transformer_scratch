def update(self, td):
        """Update state of ball"""
        self.sprite.last_position = self.sprite.position
        self.sprite.last_velocity = self.sprite.velocity
        if self.particle_group != None:
            self.update_particle_group(td)