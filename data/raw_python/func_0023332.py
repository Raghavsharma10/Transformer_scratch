def _as_rescale(self, get, targetbitdepth):
        """Helper used by :meth:`asRGB8` and :meth:`asRGBA8`."""

        width,height,pixels,meta = get()
        maxval = 2**meta['bitdepth'] - 1
        targetmaxval = 2**targetbitdepth - 1
        factor = float(targetmaxval) / float(maxval)
        meta['bitdepth'] = targetbitdepth
        def iterscale():
            for row in pixels:
                yield map(lambda x: int(round(x*factor)), row)
        if maxval == targetmaxval:
            return width, height, pixels, meta
        else:
            return width, height, iterscale(), meta