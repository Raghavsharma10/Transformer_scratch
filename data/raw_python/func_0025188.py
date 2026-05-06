def extend(self, step):
        """
        Adds the data from another STObject to this object.
        
        Args:
            step: another STObject being added after the current one in time.
        """
        self.timesteps.extend(step.timesteps)
        self.masks.extend(step.masks)
        self.x.extend(step.x)
        self.y.extend(step.y)
        self.i.extend(step.i)
        self.j.extend(step.j)
        self.end_time = step.end_time
        self.times = np.arange(self.start_time, self.end_time + self.step, self.step)
        self.u = np.concatenate((self.u, step.u))
        self.v = np.concatenate((self.v, step.v))
        for attr in self.attributes.keys():
            if attr in step.attributes.keys():
                self.attributes[attr].extend(step.attributes[attr])