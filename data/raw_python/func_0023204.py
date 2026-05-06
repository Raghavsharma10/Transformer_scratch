def viewbox_mouse_event(self, event):
        """ViewBox mouse event handler

        Parameters
        ----------
        event : instance of Event
            The mouse event.
        """
        # When the attached ViewBox reseives a mouse event, it is sent to the
        # camera here.
        
        self.mouse_pos = event.pos[:2]
        if event.type == 'mouse_wheel':
            # wheel rolled; adjust the magnification factor and hide the 
            # event from the superclass
            m = self.mag_target 
            m *= 1.2 ** event.delta[1]
            m = m if m > 1 else 1
            self.mag_target = m
        else:
            # send everything _except_ wheel events to the superclass
            super(MagnifyCamera, self).viewbox_mouse_event(event)
            
        # start the timer to smoothly modify the transform properties. 
        if not self.timer.running:
            self.timer.start()
        
        self._update_transform()