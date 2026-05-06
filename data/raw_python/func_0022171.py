def _luminance(self, rgb):
        """
        Determine the liminanace of an RGB colour
        """
        a = []
        for v in rgb:
            v = v / float(255)
            if v < 0.03928:
                result = v / 12.92
            else:
                result = math.pow(((v + 0.055) / 1.055), 2.4)

            a.append(result)
        return a[0] * 0.2126 + a[1] * 0.7152 + a[2] * 0.0722