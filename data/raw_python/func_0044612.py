def convert_svg_transform(self, transform):
        """
        Converts a string representing a SVG transform into
        AffineTransform fields.
        See https://www.w3.org/TR/SVG/coords.html#TransformAttribute for the
        specification of the transform strings. skewX and skewY are not
        supported.
        Raises:
            ValueError: If transform is not a valid and supported SVG
            transform.
        """

        tr, args = transform[:-1].split('(')
        a = map(float, args.split(' '))

        # Handle various string tranformations
        if tr == 'matrix':
            pass
        elif tr == 'translate':
            a = [1.0, 0.0, 0.0, 1.0, a[0], a[1] if len(a) > 1 else 0.0]
        elif tr == 'scale':
            a = [a[0], 0.0, 0.0, a[-1], 0.0, 0.0]
        elif tr == 'rotate':
            x = a[1] if len(a) > 1 else 0.0
            y = a[2] if len(a) > 1 else 0.0
            rad = radians(a[0])
            s = sin(rad)
            c = cos(rad)
            a = [
                c,
                s,
                -s,
                c,
                x * (1 - c) + y * s,
                -x * s + y * (1 - c),
            ]
        else:
            raise ValueError('Unknown transformation "%s"' % transform)

        self._svg_transform = transform
        self._a00 = a[0]
        self._a10 = a[1]
        self._a01 = a[2]
        self._a11 = a[3]
        self._a02 = a[4]
        self._a12 = a[5]