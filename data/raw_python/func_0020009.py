def _draw_frame(self, framedata):
        """Reads, processes and draws the frames.

        If needed for color maps, conversions to gray scale are performed. In
        case the images are no color images and no custom color maps are
        defined, the colormap `gray` is applied.

        This function is called by TimedAnimation.

        Args:
            framedata: The frame data.
        """
        original = self.read_frame()
        if original is None:
            self.update_info(self.info_string(message='Finished.',
                                              frame=framedata))
            return

        if self.original is not None:
            processed = self.process_frame(original.copy())

            if self.cmap_original is not None:
                original = to_gray(original)
            elif not is_color_image(original):
                self.original.set_cmap('gray')
            self.original.set_data(original)
        else:
            processed = self.process_frame(original)

        if self.cmap_processed is not None:
            processed = to_gray(processed)
        elif not is_color_image(processed):
            self.processed.set_cmap('gray')

        if self.annotations:
            self.annotate(framedata)

        self.processed.set_data(processed)

        self.update_info(self.info_string(frame=framedata))