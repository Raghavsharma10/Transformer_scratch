def print_info(self, capture):
        """Prints information about the unprocessed image.

        Reads one frame from the source to determine image colors, dimensions
        and data types.

        Args:
            capture: the source to read from.
        """
        self.frame_offset += 1
        ret, frame = capture.read()
        if ret:
            print('Capture Information')
            print('\tDimensions (HxW): {}x{}'.format(*frame.shape[0:2]))
            print('\tColor channels:   {}'.format(frame.shape[2] if
                                                  len(frame.shape) > 2 else 1))
            print('\tColor range:      {}-{}'.format(np.min(frame),
                                                     np.max(frame)))
            print('\tdtype:            {}'.format(frame.dtype))
        else:
            print('No source found.')