def interp(self, date: timetools.Date) -> float:
        """Perform a linear value interpolation for the given `date` and
        return the result.

        Instantiate a 1-dimensional |SeasonalParameter| object:

        >>> from hydpy.core.parametertools import SeasonalParameter
        >>> class Par(SeasonalParameter):
        ...     NDIM = 1
        ...     TYPE = float
        ...     TIME = None
        >>> par = Par(None)
        >>> par.simulationstep = '1d'
        >>> par.shape = (None,)

        Define three toy-value pairs:

        >>> par(_1=2.0, _2=5.0, _12_31=4.0)

        Passing a |Date| object matching a |TOY| object exactly returns
        the corresponding |float| value:

        >>> from hydpy import Date
        >>> par.interp(Date('2000.01.01'))
        2.0
        >>> par.interp(Date('2000.02.01'))
        5.0
        >>> par.interp(Date('2000.12.31'))
        4.0

        For all intermediate points, |SeasonalParameter.interp| performs
        a linear interpolation:

        >>> from hydpy import round_
        >>> round_(par.interp(Date('2000.01.02')))
        2.096774
        >>> round_(par.interp(Date('2000.01.31')))
        4.903226
        >>> round_(par.interp(Date('2000.02.02')))
        4.997006
        >>> round_(par.interp(Date('2000.12.30')))
        4.002994

        Linear interpolation is also allowed between the first and the
        last pair when they do not capture the endpoints of the year:

        >>> par(_1_2=2.0, _12_30=4.0)
        >>> round_(par.interp(Date('2000.12.29')))
        3.99449
        >>> par.interp(Date('2000.12.30'))
        4.0
        >>> round_(par.interp(Date('2000.12.31')))
        3.333333
        >>> round_(par.interp(Date('2000.01.01')))
        2.666667
        >>> par.interp(Date('2000.01.02'))
        2.0
        >>> round_(par.interp(Date('2000.01.03')))
        2.00551

        The following example briefly shows interpolation performed for
        a 2-dimensional parameter:

        >>> Par.NDIM = 2
        >>> par = Par(None)
        >>> par.shape = (None, 2)
        >>> par(_1_1=[1., 2.], _1_3=[-3, 0.])
        >>> result = par.interp(Date('2000.01.02'))
        >>> round_(result[0])
        -1.0
        >>> round_(result[1])
        1.0
        """
        xnew = timetools.TOY(date)
        xys = list(self)
        for idx, (x_1, y_1) in enumerate(xys):
            if x_1 > xnew:
                x_0, y_0 = xys[idx-1]
                break
        else:
            x_0, y_0 = xys[-1]
            x_1, y_1 = xys[0]
        return y_0+(y_1-y_0)/(x_1-x_0)*(xnew-x_0)