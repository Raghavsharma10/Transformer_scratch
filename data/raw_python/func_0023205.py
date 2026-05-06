def on_timer(self, event=None):
        """Timer event handler

        Parameters
        ----------
        event : instance of Event
            The timer event.
        """
        # Smoothly update center and magnification properties of the transform
        k = np.clip(100. / self.mag.mag, 10, 100)
        s = 10**(-k * event.dt)
            
        c = np.array(self.mag.center)
        c1 = c * s + self.mouse_pos * (1-s)
        
        m = self.mag.mag * s + self.mag_target * (1-s)
        
        # If changes are very small, then it is safe to stop the timer.
        if (np.all(np.abs((c - c1) / c1) < 1e-5) and 
                (np.abs(np.log(m / self.mag.mag)) < 1e-3)):
            self.timer.stop()
            
        self.mag.center = c1
        self.mag.mag = m
            
        self._update_transform()