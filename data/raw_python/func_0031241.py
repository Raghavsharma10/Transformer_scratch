def get_cap_frames(self, *frame_nums):
        """ Stop capture on ports.

        :param frame_nums: list of frame numbers to read.
        :return: list of captured frames.
        """

        frames = []
        for frame_num in frame_nums:
            if self.captureBuffer.getframe(frame_num) == '0':
                frames.append(self.captureBuffer.frame)
            else:
                frames.append(None)
        return frames