def print_time(self, message="Time is now: ", print_frame_info=True):
        """
        Print the current elapsed time.

        Kwargs:
            message (str) : Message to prefix the time stamp.
            print_frame_info (bool) : Add frame info to the print message.

        """
        if print_frame_info:
            frame_info = inspect.getouterframes(inspect.currentframe())[1]
            print(message, (datetime.now() - self.start_time), frame_info)
        else:
            print(message, (datetime.now() - self.start_time))