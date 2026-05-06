def asRGBA(self):
        """Return image as RGBA pixels.  Greyscales are expanded into
        RGB triplets; an alpha channel is synthesized if necessary.
        The return values are as for the :meth:`read` method
        except that the *metadata* reflect the returned pixels, not the
        source image.  In particular, for this method
        ``metadata['greyscale']`` will be ``False``, and
        ``metadata['alpha']`` will be ``True``.
        """

        width,height,pixels,meta = self.asDirect()
        if meta['alpha'] and not meta['greyscale']:
            return width,height,pixels,meta
        typecode = 'BH'[meta['bitdepth'] > 8]
        maxval = 2**meta['bitdepth'] - 1
        maxbuffer = struct.pack('=' + typecode, maxval) * 4 * width
        def newarray():
            return array(typecode, maxbuffer)

        if meta['alpha'] and meta['greyscale']:
            # LA to RGBA
            def convert():
                for row in pixels:
                    # Create a fresh target row, then copy L channel
                    # into first three target channels, and A channel
                    # into fourth channel.
                    a = newarray()
                    pngfilters.convert_la_to_rgba(row, a)
                    yield a
        elif meta['greyscale']:
            # L to RGBA
            def convert():
                for row in pixels:
                    a = newarray()
                    pngfilters.convert_l_to_rgba(row, a)
                    yield a
        else:
            assert not meta['alpha'] and not meta['greyscale']
            # RGB to RGBA
            def convert():
                for row in pixels:
                    a = newarray()
                    pngfilters.convert_rgb_to_rgba(row, a)
                    yield a
        meta['alpha'] = True
        meta['greyscale'] = False
        return width,height,convert(),meta