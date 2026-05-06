def run_simulation(projectname: str, xmlfile: str):
    """Perform a HydPy workflow in agreement with the given XML configuration
    file available in the directory of the given project. ToDo

    Function |run_simulation| is a "script function" and is normally used as
    explained in the main documentation on module |xmltools|.
    """
    write = commandtools.print_textandtime
    hydpy.pub.options.printprogress = False
    write(f'Start HydPy project `{projectname}`')
    hp = hydpytools.HydPy(projectname)
    write(f'Read configuration file `{xmlfile}`')
    interface = XMLInterface(xmlfile)
    write('Interpret the defined options')
    interface.update_options()
    hydpy.pub.options.printprogress = False
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
    interface.series_io.load_series()
    write('Perform the simulation run')
    hp.doit()
    write('Write the desired condition files')
    interface.conditions_io.save_conditions()
    write('Write the desired time series files')
    interface.series_io.save_series()