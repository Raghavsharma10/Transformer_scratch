def defilter(cur, prev, filter_type, bpp=4):
        """Decode a chunk"""
        if filter_type == 0:  # No filter
            return cur
        elif filter_type == 1:  # Sub
            xp = 0
            for xc in range(bpp, len(cur)):
                cur[xc] = (cur[xc] + cur[xp]) % 256
                xp += 1
        elif filter_type == 2:  # Up
            for xc in range(len(cur)):
                cur[xc] = (cur[xc] + prev[xc]) % 256
        elif filter_type == 3:  # Average
            xp = 0
            for i in range(bpp):
                cur[i] = (cur[i] + prev[i] // 2) % 256
            for xc in range(bpp, len(cur)):
                cur[xc] = (cur[xc] + ((cur[xp] + prev[xc]) // 2)) % 256
                xp += 1
        elif filter_type == 4:  # Paeth
            xp = 0
            for i in range(bpp):
                cur[i] = (cur[i] + prev[i]) % 256
            for xc in range(bpp, len(cur)):
                a = cur[xp]
                b = prev[xc]
                c = prev[xp]
                p = a + b - c
                pa = abs(p - a)
                pb = abs(p - b)
                pc = abs(p - c)
                if pa <= pb and pa <= pc:
                    value = a
                elif pb <= pc:
                    value = b
                else:
                    value = c
                cur[xc] = (cur[xc] + value) % 256
                xp += 1
        else:
            raise ValueError('Unrecognized scanline filter type: {}'.format(filter_type))
        return cur