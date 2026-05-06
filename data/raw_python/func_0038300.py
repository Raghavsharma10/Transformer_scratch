def set_bumper_color(self, particle, group, bumper, collision_point, collision_normal):
		"""Set bumper color to the color of the particle that collided with it"""
		self.color = tuple(particle.color)[:3]