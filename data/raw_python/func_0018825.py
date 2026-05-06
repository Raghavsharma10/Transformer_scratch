def initialise(self, projectname: str, xmlfile: str) -> None:
        """Initialise a *HydPy* project based on the given XML configuration
        file agreeing with `HydPyConfigMultipleRuns.xsd`.

        We use the `LahnH` project and its rather complex XML configuration
        file `multiple_runs.xml` as an example (module |xmltools| provides
        information on interpreting this file):

        >>> from hydpy.core.examples import prepare_full_example_1
        >>> prepare_full_example_1()
        >>> from hydpy import print_values, TestIO
        >>> from hydpy.exe.servertools import ServerState
        >>> state = ServerState()
        >>> with TestIO():    # doctest: +ELLIPSIS
        ...     state.initialise('LahnH', 'multiple_runs.xml')
        Start HydPy project `LahnH` (...).
        Read configuration file `multiple_runs.xml` (...).
        Interpret the defined options (...).
        Interpret the defined period (...).
        Read all network files (...).
        Activate the selected network (...).
        Read the required control files (...).
        Read the required condition files (...).
        Read the required time series files (...).

        After initialisation, all defined exchange items are available:

        >>> for item in state.parameteritems:
        ...     print(item)
        SetItem('alpha', 'hland_v1', 'control.alpha', 0)
        SetItem('beta', 'hland_v1', 'control.beta', 0)
        SetItem('lag', 'hstream_v1', 'control.lag', 0)
        SetItem('damp', 'hstream_v1', 'control.damp', 0)
        AddItem('sfcf_1', 'hland_v1', 'control.sfcf', 'control.rfcf', 0)
        AddItem('sfcf_2', 'hland_v1', 'control.sfcf', 'control.rfcf', 0)
        AddItem('sfcf_3', 'hland_v1', 'control.sfcf', 'control.rfcf', 1)
        >>> for item in state.conditionitems:
        ...     print(item)
        SetItem('sm_lahn_2', 'hland_v1', 'states.sm', 0)
        SetItem('sm_lahn_1', 'hland_v1', 'states.sm', 1)
        SetItem('quh', 'hland_v1', 'logs.quh', 0)
        >>> for item in state.getitems:
        ...     print(item)
        GetItem('hland_v1', 'fluxes.qt')
        GetItem('hland_v1', 'fluxes.qt.series')
        GetItem('hland_v1', 'states.sm')
        GetItem('hland_v1', 'states.sm.series')
        GetItem('nodes', 'nodes.sim.series')

        The initialisation also memorises the initial conditions of
        all elements:

        >>> for element in state.init_conditions:
        ...     print(element)
        land_dill
        land_lahn_1
        land_lahn_2
        land_lahn_3
        stream_dill_lahn_2
        stream_lahn_1_lahn_2
        stream_lahn_2_lahn_3

        Initialisation also prepares all selected series arrays and
        reads the required input data:

        >>> print_values(
        ...     state.hp.elements.land_dill.model.sequences.inputs.t.series)
        -0.298846, -0.811539, -2.493848, -5.968849, -6.999618
        >>> state.hp.nodes.dill.sequences.sim.series
        InfoArray([ nan,  nan,  nan,  nan,  nan])
        """
        write = commandtools.print_textandtime
        write(f'Start HydPy project `{projectname}`')
        hp = hydpytools.HydPy(projectname)
        write(f'Read configuration file `{xmlfile}`')
        interface = xmltools.XMLInterface(xmlfile)
        write('Interpret the defined options')
        interface.update_options()
        write('Interpret the defined period')
        interface.update_timegrids()
        write('Read all network files')
        hp.prepare_network()
        write('Activate the selected network')
        hp.update_devices(interface.fullselection)
        write('Read the required control files')
        hp.init_models()
        write('Read the required condition files')
        interface.conditions_io.load_conditions()
        write('Read the required time series files')
        interface.series_io.prepare_series()
        interface.exchange.prepare_series()
        interface.series_io.load_series()
        self.hp = hp
        self.parameteritems = interface.exchange.parameteritems
        self.conditionitems = interface.exchange.conditionitems
        self.getitems = interface.exchange.getitems
        self.conditions = {}
        self.parameteritemvalues = collections.defaultdict(lambda: {})
        self.modifiedconditionitemvalues = collections.defaultdict(lambda: {})
        self.getitemvalues = collections.defaultdict(lambda: {})
        self.init_conditions = hp.conditions
        self.timegrids = {}