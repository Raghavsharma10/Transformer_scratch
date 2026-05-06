def ansi_format_iter( self, x_start=0, y_start=0, width=None, height=None, frame=0, columns=1, downsample=1, frame_index=None, frame_flip_v=0, frame_flip_h=0 ):
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
            Single frame number/object, or a list of frames to render in sequence. Defaults to frame 0.

        columns
            Number of frames to render per line (useful for printing tilemaps!). Defaults to 1.

        downsample
            Shrink larger images by printing every nth pixel only. Defaults to 1.

        frame_index
            Constant or mrc.Ref for a frame object property denoting the index. Defaults to None
            (i.e. frame itself should be an index).

        frame_flip_v
            Constant or mrc.Ref for a frame object property for whether to mirror vertically.
            Defaults to 0.

        frame_flip_h
            Constant or mrc.Ref for a frame object property for whether to mirror horizontally.
            Defaults to 0.
        """

        assert x_start in range( 0, self.width )
        assert y_start in range( 0, self.height )
        if frame_index is not None:
            fn_index = lambda fr: mrc.property_get( frame_index, fr )
        else:
            fn_index = lambda fr: fr if fr in range( 0, self.frame_count ) else None
        fn_flip_v = lambda fr: mrc.property_get( frame_flip_v, fr )
        fn_flip_h = lambda fr: mrc.property_get( frame_flip_h, fr )

        frames = []
        try:
            frame_iter = iter( frame )
            frames = [f for f in frame_iter]
        except TypeError:
            frames = [frame]

        if not width:
            width = self.width-x_start
        if not height:
            height = self.height-y_start

        stride = width*height

        def data_fetch( x, y, fr_obj ):
            fr = fn_index( fr_obj )
            if fr is None:
                return Transparent()
            if not ((0 <= x < self.width) and (0 <= y < self.height)):
                return Transparent()
            if fn_flip_h( fr_obj ):
                x = self.width - x - 1
            if fn_flip_v( fr_obj ):
                y = self.height - y - 1
            index = self.width*y + x
            p = self.source[stride*fr+index]
            if self.mask:
                p = p if self.mask[stride*fr+index] else None
            return self.palette[p] if p is not None else Transparent()

        for x in ansi.format_image_iter( data_fetch, x_start, y_start, width, height, frames, columns, downsample ):
            yield x
        return