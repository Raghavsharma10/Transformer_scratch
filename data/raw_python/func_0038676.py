def update(self, time_delta):
        """Update all sprites in the system. time_delta is the
        time since the last update (in arbitrary time units).

        This method can be conveniently scheduled using the Pyglet
        scheduler method: pyglet.clock.schedule_interval
        """
        self.control.update(self, time_delta)
        for object in self.objects:
            object.update(time_delta)
            # object.sprite.last_position = object.sprite.position
            # object.sprite.last_velocity = object.sprite.velocity

        # for group in self:
        for controller in self.controllers:
            controller(time_delta, self)