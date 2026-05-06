def prepare_sequencemanager(self) -> None:
        """Configure the |SequenceManager| object available in module
        |pub| following the definitions of the actual XML `reader` or
        `writer` element when available; if not use those of the XML
        `series_io` element.

        Compare the following results with `single_run.xml` to see that the
        first `writer` element defines the input file type specifically,
        that the second `writer` element defines a general file type, and
        that the third `writer` element does not define any file type (the
        principle mechanism is the same for other options, e.g. the
        aggregation mode):

        >>> from hydpy.core.examples import prepare_full_example_1
        >>> prepare_full_example_1()

        >>> from hydpy import HydPy, TestIO, XMLInterface, pub
        >>> hp = HydPy('LahnH')
        >>> with TestIO():
        ...     hp.prepare_network()
        ...     interface = XMLInterface('single_run.xml')
        >>> series_io = interface.series_io
        >>> with TestIO():
        ...     series_io.writers[0].prepare_sequencemanager()
        >>> pub.sequencemanager.inputfiletype
        'asc'
        >>> pub.sequencemanager.fluxfiletype
        'npy'
        >>> pub.sequencemanager.fluxaggregation
        'none'
        >>> with TestIO():
        ...     series_io.writers[1].prepare_sequencemanager()
        >>> pub.sequencemanager.statefiletype
        'nc'
        >>> pub.sequencemanager.stateoverwrite
        False
        >>> with TestIO():
        ...     series_io.writers[2].prepare_sequencemanager()
        >>> pub.sequencemanager.statefiletype
        'npy'
        >>> pub.sequencemanager.fluxaggregation
        'mean'
        >>> pub.sequencemanager.inputoverwrite
        True
        >>> pub.sequencemanager.inputdirpath
        'LahnH/series/input'
        """
        for config, convert in (
                ('filetype', lambda x: x),
                ('aggregation', lambda x: x),
                ('overwrite', lambda x: x.lower() == 'true'),
                ('dirpath', lambda x: x)):
            xml_special = self.find(config)
            xml_general = self.master.find(config)
            for name_manager, name_xml in zip(
                    ('input', 'flux', 'state', 'node'),
                    ('inputs', 'fluxes', 'states', 'nodes')):
                value = None
                for xml, attr_xml in zip(
                        (xml_special, xml_special, xml_general, xml_general),
                        (name_xml, 'general', name_xml, 'general')):
                    try:
                        value = find(xml, attr_xml).text
                    except AttributeError:
                        continue
                    break
                setattr(hydpy.pub.sequencemanager,
                        f'{name_manager}{config}',
                        convert(value))