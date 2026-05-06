def update_timegrids(self) -> None:
        """Update the |Timegrids| object available in module |pub| with the
        values defined in the `timegrid` XML element.

        Usually, one would prefer to define `firstdate`, `lastdate`, and
        `stepsize` elements as in the XML configuration file of the
        `LahnH` example project:

        >>> from hydpy.core.examples import prepare_full_example_1
        >>> prepare_full_example_1()
        >>> from hydpy import HydPy, pub, TestIO
        >>> from hydpy.auxs.xmltools import XMLInterface

        >>> hp = HydPy('LahnH')
        >>> with TestIO():
        ...     hp.prepare_network()
        ...     XMLInterface('single_run.xml').update_timegrids()
        >>> pub.timegrids
        Timegrids(Timegrid('1996-01-01T00:00:00',
                           '1996-01-06T00:00:00',
                           '1d'))

        Alternatively, one can provide the file path to a `seriesfile`,
        which must be a valid NetCDF file.  The |XMLInterface| object
        then interprets the file's time information:

        >>> name = 'LahnH/series/input/hland_v1_input_p.nc'
        >>> with TestIO():
        ...     with open('LahnH/single_run.xml') as file_:
        ...         lines = file_.readlines()
        ...     for idx, line in enumerate(lines):
        ...         if '<timegrid>' in line:
        ...             break
        ...     with open('LahnH/single_run.xml', 'w') as file_:
        ...         _ = file_.write(''.join(lines[:idx+1]))
        ...         _ = file_.write(
        ...             f'        <seriesfile>{name}</seriesfile>\\n')
        ...         _ = file_.write(''.join(lines[idx+4:]))
        ...     XMLInterface('single_run.xml').update_timegrids()
        >>> pub.timegrids
        Timegrids(Timegrid('1996-01-01 00:00:00',
                           '2007-01-01 00:00:00',
                           '1d'))
        """
        timegrid_xml = self.find('timegrid')
        try:
            timegrid = timetools.Timegrid(
                *(timegrid_xml[idx].text for idx in range(3)))
            hydpy.pub.timegrids = timetools.Timegrids(timegrid)
        except IndexError:
            seriesfile = find(timegrid_xml, 'seriesfile').text
            with netcdf4.Dataset(seriesfile) as ncfile:
                hydpy.pub.timegrids = timetools.Timegrids(
                    netcdftools.query_timegrid(ncfile))