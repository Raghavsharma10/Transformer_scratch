def make_frame(self, frame, birthframe, startframe, stopframe, deathframe, noiseframe=None):
        """
        :param frame: current frame 
        :param birthframe: frame where this animation starts returning something other than None
        :param startframe: frame where animation starts to evolve
        :param stopframe: frame where animation is completed
        :param deathframe: frame where animation starts to return None
        :return: 
        """
        return self.anim.make_frame(frame, birthframe, startframe, stopframe, deathframe, noiseframe)