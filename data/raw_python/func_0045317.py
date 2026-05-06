def make_frame(self, frame, birthframe, startframe, stopframe, deathframe, noiseframe=None):
        """
        :param frame: current frame 
        :param birthframe: frame where animation starts to return something other than None
        :param startframe: frame where animation starts to evolve
        :param stopframe: frame where animation stops evolving
        :param deathframe: frame where animation starts to return None
        :return: 
        """
        if self.use_alpha:
            return (self.__clip(self.anim_red.make_frame(frame, birthframe, startframe, stopframe, deathframe, noiseframe), 0, 1),
                    self.__clip(self.anim_green.make_frame(frame, birthframe, startframe, stopframe, deathframe, noiseframe), 0, 1),
                    self.__clip(self.anim_blue.make_frame(frame, birthframe, startframe, stopframe, deathframe, noiseframe), 0, 1))
        else:
            return (self.__clip(self.anim_red.make_frame(frame, birthframe, startframe, stopframe, deathframe, noiseframe), 0, 1),
                    self.__clip(self.anim_green.make_frame(frame, birthframe, startframe, stopframe, deathframe, noiseframe), 0, 1),
                    self.__clip(self.anim_blue.make_frame(frame, birthframe, startframe, stopframe, deathframe, noiseframe), 0, 1),
                    self.__clip(self.anim_alpha.make_frame(frame, birthframe, startframe, stopframe, deathframe, noiseframe), 0, 1))