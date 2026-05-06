def handle_pause(self):
        """Read pause signal from server"""
        flag = self.reader.byte()
        if flag > 0:
            logger.info(" -> pause: on")
            self.controller.playing = False
        else:
            logger.info(" -> pause: off")
            self.controller.playing = True