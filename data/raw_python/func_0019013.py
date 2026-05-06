def validate_xml(self) -> None:
        """Raise an error if the actual XML does not agree with one of the
        available schema files.

        # ToDo: should it be accompanied by a script function?

        The first example relies on a distorted version of the configuration
        file `single_run.xml`:

        >>> from hydpy.core.examples import prepare_full_example_1
        >>> prepare_full_example_1()
        >>> from hydpy import TestIO, xml_replace
        >>> from hydpy.auxs.xmltools import XMLInterface
        >>> import os
        >>> with TestIO():    # doctest: +ELLIPSIS
        ...     xml_replace('LahnH/single_run',
        ...                 firstdate='1996-01-32T00:00:00')
        template file: LahnH/single_run.xmlt
        target file: LahnH/single_run.xml
        replacements:
          config_start --> <...HydPyConfigBase.xsd"
                      ...HydPyConfigSingleRun.xsd"> (default argument)
          firstdate --> 1996-01-32T00:00:00 (given argument)
          zip_ --> false (default argument)
          zip_ --> false (default argument)
          config_end --> </hpcsr:config> (default argument)
        >>> with TestIO():
        ...     interface = XMLInterface('single_run.xml', 'LahnH')
        >>> interface.validate_xml()    # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        hydpy.core.objecttools.xmlschema.validators.exceptions.\
XMLSchemaDecodeError: While trying to validate XML file `...single_run.xml`, \
the following error occurred: failed validating '1996-01-32T00:00:00' with \
XsdAtomicBuiltin(name='xs:dateTime').
        ...
        Reason: day is out of range for month
        ...
        Schema:
        ...
        Instance:
        ...
          <firstdate xmlns="https://github.com/hydpy-dev/hydpy/releases/\
download/your-hydpy-version/HydPyConfigBase.xsd">1996-01-32T00:00:00</firstdate>
        ...
        Path: /hpcsr:config/timegrid/firstdate
        ...

        In the second example, we examine a correct configuration file:

        >>> with TestIO():    # doctest: +ELLIPSIS
        ...     xml_replace('LahnH/single_run')
        ...     interface = XMLInterface('single_run.xml', 'LahnH')
        template file: LahnH/single_run.xmlt
        target file: LahnH/single_run.xml
        replacements:
          config_start --> <...HydPyConfigBase.xsd"
                      ...HydPyConfigSingleRun.xsd"> (default argument)
          firstdate --> 1996-01-01T00:00:00 (default argument)
          zip_ --> false (default argument)
          zip_ --> false (default argument)
          config_end --> </hpcsr:config> (default argument)
        >>> interface.validate_xml()

        The XML configuration file must correctly refer to the corresponding
        schema file:

        >>> with TestIO():    # doctest: +ELLIPSIS
        ...     xml_replace('LahnH/single_run',
        ...                 config_start='<config>',
        ...                 config_end='</config>')
        ...     interface = XMLInterface('single_run.xml', 'LahnH')
        template file: LahnH/single_run.xmlt
        target file: LahnH/single_run.xml
        replacements:
          config_start --> <config> (given argument)
          firstdate --> 1996-01-01T00:00:00 (default argument)
          zip_ --> false (default argument)
          zip_ --> false (default argument)
          config_end --> </config> (given argument)
        >>> interface.validate_xml()    # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        RuntimeError: While trying to validate XML file `...single_run.xml`, \
the following error occurred: Configuration file `single_run.xml` does not \
correctly refer to one of the available XML schema files \
(HydPyConfigSingleRun.xsd and HydPyConfigMultipleRuns.xsd).

        XML files based on `HydPyConfigMultipleRuns.xsd` can be validated
        as well:

        >>> with TestIO():
        ...     interface = XMLInterface('multiple_runs.xml', 'LahnH')
        >>> interface.validate_xml()    # doctest: +ELLIPSIS
        """
        try:
            filenames = ('HydPyConfigSingleRun.xsd',
                         'HydPyConfigMultipleRuns.xsd')
            for name in filenames:
                if name in self.root.tag:
                    schemafile = name
                    break
            else:
                raise RuntimeError(
                    f'Configuration file `{os.path.split(self.filepath)[-1]}` '
                    f'does not correctly refer to one of the available XML '
                    f'schema files ({objecttools.enumeration(filenames)}).')
            schemapath = os.path.join(conf.__path__[0], schemafile)
            schema = xmlschema.XMLSchema(schemapath)
            schema.validate(self.filepath)
        except BaseException:
            objecttools.augment_excmessage(
                f'While trying to validate XML file `{self.filepath}`')