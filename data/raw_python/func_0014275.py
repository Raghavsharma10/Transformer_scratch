def statexml2pdb(topology, state, output=None):
    """
    Given an OpenMM xml file containing the state of the simulation,
    generate a PDB snapshot for easy visualization.
    """
    state = Restart.from_xml(state)
    system = SystemHandler.load(topology, positions=state.positions)
    if output is None:
        output = topology + '.pdb'
    system.write_pdb(output)