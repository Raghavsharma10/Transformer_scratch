def bokeh_draw_court(figure, line_color='gray', line_width=1):
    """Returns a figure with the basketball court lines drawn onto it

    This function draws a court based on the x and y-axis values that the NBA
    stats API provides for the shot chart data.  For example the center of the
    hoop is located at the (0,0) coordinate.  Twenty-two feet from the left of
    the center of the hoop in is represented by the (-220,0) coordinates.
    So one foot equals +/-10 units on the x and y-axis.

    Parameters
    ----------
    figure : Bokeh figure object
        The Axes object to plot the court onto.
    line_color : str, optional
        The color of the court lines. Can be a a Hex value.
    line_width : float, optional
        The linewidth the of the court lines in pixels.

    Returns
    -------
    figure : Figure
        The Figure object with the court on it.

    """

    # hoop
    figure.circle(x=0, y=0, radius=7.5, fill_alpha=0,
                  line_color=line_color, line_width=line_width)

    # backboard
    figure.line(x=range(-30, 31), y=-12.5, line_color=line_color)

    # The paint
    # outerbox
    figure.rect(x=0, y=47.5, width=160, height=190, fill_alpha=0,
                line_color=line_color, line_width=line_width)
    # innerbox
    # left inner box line
    figure.line(x=-60, y=np.arange(-47.5, 143.5), line_color=line_color,
                line_width=line_width)
    # right inner box line
    figure.line(x=60, y=np.arange(-47.5, 143.5), line_color=line_color,
                line_width=line_width)

    # Restricted Zone
    figure.arc(x=0, y=0, radius=40, start_angle=pi, end_angle=0,
               line_color=line_color, line_width=line_width)

    # top free throw arc
    figure.arc(x=0, y=142.5, radius=60, start_angle=pi, end_angle=0,
               line_color=line_color)

    # bottome free throw arc
    figure.arc(x=0, y=142.5, radius=60, start_angle=0, end_angle=pi,
               line_color=line_color, line_dash="dashed")

    # Three point line
    # corner three point lines
    figure.line(x=-220, y=np.arange(-47.5, 92.5), line_color=line_color,
                line_width=line_width)
    figure.line(x=220, y=np.arange(-47.5, 92.5), line_color=line_color,
                line_width=line_width)
    # # three point arc
    figure.arc(x=0, y=0, radius=237.5, start_angle=3.528, end_angle=-0.3863,
               line_color=line_color, line_width=line_width)

    # add center court
    # outer center arc
    figure.arc(x=0, y=422.5, radius=60, start_angle=0, end_angle=pi,
               line_color=line_color, line_width=line_width)
    # inner center arct
    figure.arc(x=0, y=422.5, radius=20, start_angle=0, end_angle=pi,
               line_color=line_color, line_width=line_width)

    # outer lines, consistting of half court lines and out of bounds lines
    figure.rect(x=0, y=187.5, width=500, height=470, fill_alpha=0,
                line_color=line_color, line_width=line_width)

    return figure