def write_xsd(cls) -> None:
        """Write the complete base schema file `HydPyConfigBase.xsd` based
        on the template file `HydPyConfigBase.xsdt`.

        Method |XSDWriter.write_xsd| adds model specific information to the
        general information of template file `HydPyConfigBase.xsdt` regarding
        reading and writing of time series data and exchanging parameter
        and sequence values e.g. during calibration.

        The following example shows that after writing a new schema file,
        method |XMLInterface.validate_xml| does not raise an error when
        either applied on the XML configuration files `single_run.xml` or
        `multiple_runs.xml` of the `LahnH` example project:

        >>> import os
        >>> from hydpy.auxs.xmltools import XSDWriter, XMLInterface
        >>> if os.path.exists(XSDWriter.filepath_target):
        ...     os.remove(XSDWriter.filepath_target)
        >>> os.path.exists(XSDWriter.filepath_target)
        False
        >>> XSDWriter.write_xsd()
        >>> os.path.exists(XSDWriter.filepath_target)
        True

        >>> from hydpy import data
        >>> for configfile in ('single_run.xml', 'multiple_runs.xml'):
        ...     XMLInterface(configfile, data.get_path('LahnH')).validate_xml()
        """
        with open(cls.filepath_source) as file_:
            template = file_.read()
        template = template.replace(
            '<!--include model sequence groups-->', cls.get_insertion())
        template = template.replace(
            '<!--include exchange items-->', cls.get_exchangeinsertion())
        with open(cls.filepath_target, 'w') as file_:
            file_.write(template)