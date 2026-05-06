def curve_points(self, beginframe, endframe, framestep, birthframe, startframe, stopframe, deathframe,
                     filternone=False, noiseframe=None):
        """
        returns a list of frames from startframe to stopframe, in steps of framestep
        :param beginframe: first frame to include in list of points
        :param endframe: last frame to include in list of points
        :param framestep: framestep, e.g. 0.01 means that the points will be calculated in timesteps of 0.01
        :param birthframe: frame before which animation always returns None
        :param startframe: frame from which animation starts to evolve 
        :param stopframe: frame in which animation completed
        :param deathframe: frame in which animation starts returning None 
        :param filternone: removes all "None" values from the list of curve_points. The reply can still be None if no curve_points
        can be calculated.
        :param noiseframe: for time varying noise, this frame represents the current time for which the noise 
        must be evaluated
        :return: list of tweened values
        """
        return None