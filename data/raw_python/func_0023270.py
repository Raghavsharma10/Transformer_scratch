def _update(self):
        """Rebuilds the shaders, and repositions the objects
           that are used internally by the ColorBarVisual
        """

        x, y = self._pos
        halfw, halfh = self._halfdim

        # test that width and height are non-zero
        if halfw <= 0:
            raise ValueError("half-width must be positive and non-zero"
                             ", not %s" % halfw)
        if halfh <= 0:
            raise ValueError("half-height must be positive and non-zero"
                             ", not %s" % halfh)

        # test that the given width and height is consistent
        # with the orientation
        if (self._orientation == "bottom" or self._orientation == "top"):
                if halfw < halfh:
                    raise ValueError("half-width(%s) < half-height(%s) for"
                                     "%s orientation,"
                                     " expected half-width >= half-height" %
                                     (halfw, halfh, self._orientation, ))
        else:  # orientation == left or orientation == right
            if halfw > halfh:
                raise ValueError("half-width(%s) > half-height(%s) for"
                                 "%s orientation,"
                                 " expected half-width <= half-height" %
                                 (halfw, halfh, self._orientation, ))

        # Set up the attributes that the shaders require
        vertices = np.array([[x - halfw, y - halfh],
                             [x + halfw, y - halfh],
                             [x + halfw, y + halfh],
                             # tri 2
                             [x - halfw, y - halfh],
                             [x + halfw, y + halfh],
                             [x - halfw, y + halfh]],
                            dtype=np.float32)

        self.shared_program['a_position'] = vertices