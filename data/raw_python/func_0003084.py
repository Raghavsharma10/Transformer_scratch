def _raw_print_image(self, line, size, output=None ):
        """ Print formatted image """
        i = 0
        cont = 0
        buffer = ""
        raw = ""

        def __raw(string):
            if output:
                output(string)
            else:
                self._raw(string)
       
        raw += S_RASTER_N
        buffer = "%02X%02X%02X%02X" % (((size[0]/size[1])/8), 0, size[1], 0)
        raw += buffer.decode('hex')
        buffer = ""

        while i < len(line):
            hex_string = int(line[i:i+8],2)
            buffer += "%02X" % hex_string
            i += 8
            cont += 1
            if cont % 4 == 0:
                raw += buffer.decode("hex")
                buffer = ""
                cont = 0

        return raw