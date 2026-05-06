def dump(self):
        """Dump the image data"""
        scan_lines = bytearray()
        for y in range(self.height):
            scan_lines.append(0)  # filter type 0 (None)
            scan_lines.extend(
                self.canvas[(y * self.width * 4):((y + 1) * self.width * 4)]
            )
        # image represented as RGBA tuples, no interlacing
        return SIGNATURE + \
            self.pack_chunk(b'IHDR', struct.pack(b"!2I5B",
                                                 self.width, self.height,
                                                 8, 6, 0, 0, 0)) + \
            self.pack_chunk(b'IDAT', zlib.compress(bytes(scan_lines), 9)) + \
            self.pack_chunk(b'IEND', b'')