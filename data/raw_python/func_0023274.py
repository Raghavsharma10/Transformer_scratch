def size(self):
        """ The size of the ColorBar

        Returns
        -------
        size: (major_axis_length, minor_axis_length)
            major and minor axis are defined by the
            orientation of the ColorBar
        """
        (halfw, halfh) = self._halfdim
        if self.orientation in ["top", "bottom"]:
            return (halfw * 2., halfh * 2.)
        else:
            return (halfh * 2., halfw * 2.)