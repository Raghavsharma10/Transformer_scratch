def ansi_format_iter( self, x_start=0, y_start=0, width=None, height=None, frame=0, columns=1, downsample=1 ):
        """Return the ANSI escape sequence to render the image.

        x_start
            Offset from the left of the image data to render from. Defaults to 0.

        y_start
            Offset from the top of the image data to render from. Defaults to 0.

        width
            Width of the image data to render. Defaults to the image width.

        height
            Height of the image data to render. Defaults to the image height.

        frame
            Single frame number, or a list of frame numbers to render in sequence. Defaults to frame 0.

        columns
            Number of frames to render per line (useful for printing tilemaps!). Defaults to 1.

        downsample
            Shrink larger images by printing every nth pixel only. Defaults to 1.
        """

        image = self.get_image()
        frames = []
        frame_count = 1 if not hasattr( image, 'n_frames' ) else image.n_frames
        if isinstance( frame, int ):
            assert frame in range( 0, frame_count )
            frames = [frame]
        else:
            frames = [f for f in frame if f in range( 0, frame_count )]

        if not width:
            width = image.size[0]-x_start
        if not height:
            height = image.size[1]-y_start

        if image.mode == 'P':
            palette = from_palette_bytes( image.getpalette() )

            def data_fetch( x, y, fr ):
                if fr not in range( 0, frame_count ):
                    return Transparent()
                if not ((0 <= x < image.size[0]) and (0 <= y < image.size[1])):
                    return Transparent()
                image.seek( fr )
                return palette[image.getpixel( (x, y) )]

            for x in ansi.format_image_iter( data_fetch, x_start, y_start, width, height, frames, columns, downsample ):
                yield x
        return