def run_ahead(self, time, framerate):
        """Run the particle system for the specified time frame at the
        specified framerate to move time forward as quickly as possible.
        Useful for "warming up" the particle system to reach a steady-state
        before anything is drawn or to simply "skip ahead" in time.

        time -- The amount of simulation time to skip over.

        framerate -- The framerate of the simulation in updates per unit
        time. Higher values will increase simulation accuracy,
        but will take longer to compute.
        """
        if time:
            td = 1.0 / framerate
            update = self.update
            for i in range(int(time / td)):
                update(td)