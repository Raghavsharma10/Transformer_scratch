def move(self, lr, fb, vv, va):
        """Makes the drone move (translate/rotate).

        Parameters:
        lr -- left-right tilt: float [-1..1] negative: left, positive: right
        fb -- front-back tilt: float [-1..1] negative: forwards, positive:
            backwards
        vv -- vertical speed: float [-1..1] negative: go down, positive: rise
        va -- angular speed: float [-1..1] negative: spin left, positive: spin
            right"""
        self.at(ardrone.at.pcmd, True, lr, fb, vv, va)