def calc_size(rect, orientation):
        """Calculate a size

        Parameters
        ----------
        rect : rectangle
            The rectangle.
        orientation : str
            Either "bottom" or "top".
        """
        (total_halfx, total_halfy) = rect.center
        if orientation in ["bottom", "top"]:
            (total_major_axis, total_minor_axis) = (total_halfx, total_halfy)
        else:
            (total_major_axis, total_minor_axis) = (total_halfy, total_halfx)

        major_axis = total_major_axis * (1.0 -
                                         ColorBarWidget.major_axis_padding)
        minor_axis = major_axis * ColorBarWidget.minor_axis_ratio

        # if the minor axis is "leaking" from the padding, then clamp
        minor_axis = np.minimum(minor_axis,
                                total_minor_axis *
                                (1.0 - ColorBarWidget.minor_axis_padding))

        return (major_axis, minor_axis)