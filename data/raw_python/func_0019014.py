def update_options(self) -> None:
        """Update the |Options| object available in module |pub| with the
        values defined in the `options` XML element.

        >>> from hydpy.auxs.xmltools import XMLInterface
        >>> from hydpy import data, pub
        >>> interface = XMLInterface('single_run.xml', data.get_path('LahnH'))
        >>> pub.options.printprogress = True
        >>> pub.options.printincolor = True
        >>> pub.options.reprdigits = -1
        >>> pub.options.utcoffset = -60
        >>> pub.options.ellipsis = 0
        >>> pub.options.warnsimulationstep = 0
        >>> interface.update_options()
        >>> pub.options
        Options(
            autocompile -> 1
            checkseries -> 1
            dirverbose -> 0
            ellipsis -> 0
            forcecompiling -> 0
            printprogress -> 0
            printincolor -> 0
            reprcomments -> 0
            reprdigits -> 6
            skipdoctests -> 0
            trimvariables -> 1
            usecython -> 1
            usedefaultvalues -> 0
            utcoffset -> 60
            warnmissingcontrolfile -> 0
            warnmissingobsfile -> 1
            warnmissingsimfile -> 1
            warnsimulationstep -> 0
            warntrim -> 1
            flattennetcdf -> True
            isolatenetcdf -> True
            timeaxisnetcdf -> 0
        )
        >>> pub.options.printprogress = False
        >>> pub.options.reprdigits = 6
        """
        options = hydpy.pub.options
        for option in self.find('options'):
            value = option.text
            if value in ('true', 'false'):
                value = value == 'true'
            setattr(options, strip(option.tag), value)
        options.printprogress = False
        options.printincolor = False