def next_state(self):
        """This is a method that will be called when the time remaining ends.
        The current state can be: roasting, cooling, idle, sleeping, connecting,
        or unkown."""
        if(self.roaster.get_roaster_state() == 'roasting'):
            self.roaster.time_remaining = 20
            self.roaster.cool()
        elif(self.roaster.get_roaster_state() == 'cooling'):
            self.roaster.idle()