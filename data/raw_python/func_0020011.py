def info_string(self, size=None, message='', frame=-1):
        """Returns information about the stream.

        Generates a string containing size, frame number, and info messages.
        Omits unnecessary information (e.g. empty messages and frame -1).

        This method is primarily used to update the suptitle of the plot
        figure.

        Returns:
            An info string.
        """
        info = []
        if size is not None:
            info.append('Size: {1}x{0}'.format(*size))
        elif self.size is not None:
            info.append('Size: {1}x{0}'.format(*self.size))
        if frame >= 0:
            info.append('Frame: {}'.format(frame))
        if message != '':
            info.append('{}'.format(message))
        return ' '.join(info)