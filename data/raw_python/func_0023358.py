def _resize(self, shape, format=None, internalformat=None):
        """Internal method for resize.
        """
        shape = self._normalize_shape(shape)

        # Check
        if not self._resizable:
            raise RuntimeError("Texture is not resizable")

        # Determine format
        if format is None:
            format = self._formats[shape[-1]]
            # Keep current format if channels match
            if self._format and \
               self._inv_formats[self._format] == self._inv_formats[format]:
                format = self._format
        else:
            format = check_enum(format)

        if internalformat is None:
            # Keep current internalformat if channels match
            if self._internalformat and \
               self._inv_internalformats[self._internalformat] == shape[-1]:
                internalformat = self._internalformat
        else:

            internalformat = check_enum(internalformat)

        # Check
        if format not in self._inv_formats:
            raise ValueError('Invalid texture format: %r.' % format)
        elif shape[-1] != self._inv_formats[format]:
            raise ValueError('Format does not match with given shape. '
                             '(format expects %d elements, data has %d)' %
                             (self._inv_formats[format], shape[-1]))
        
        if internalformat is None:
            pass
        elif internalformat not in self._inv_internalformats:
            raise ValueError(
                'Invalid texture internalformat: %r. Allowed formats: %r' 
                % (internalformat, self._inv_internalformats)
            )
        elif shape[-1] != self._inv_internalformats[internalformat]:
            raise ValueError('Internalformat does not match with given shape.')

        # Store and send GLIR command
        self._shape = shape
        self._format = format
        self._internalformat = internalformat
        self._glir.command('SIZE', self._id, self._shape, self._format, 
                           self._internalformat)