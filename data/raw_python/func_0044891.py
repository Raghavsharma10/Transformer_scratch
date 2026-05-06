def make_frame(self, frame, birthframe, startframe, stopframe, deathframe, noiseframe=None):
        """
        :param frame: current frame 
        :param birthframe: frame where this animation starts returning something other than None
        :param startframe: frame where animation starts to evolve
        :param stopframe: frame where animation is completed
        :param deathframe: frame where animation starts to return None
        :return: 
        """
        newx = self.anim_x.make_frame(frame, birthframe, startframe, stopframe, deathframe, noiseframe)
        newy = self.anim_y.make_frame(frame, birthframe, startframe, stopframe, deathframe, noiseframe)

        if self.xy_noise_fn is not None:
            if noiseframe is not None:
                t = noiseframe
            else:
                t = Tween.tween2(frame, startframe, stopframe)
            addx, addy = self.xy_noise_fn(newx, newy, t)
        else:
            addx, addy = 0, 0
        return newx + addx, newy + addy