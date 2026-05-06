def export_frame_coordinates(topology, trajectory, nframe, output=None):
    """
    Extract a single frame structure from a trajectory.
    """
    if output is None:
        basename, ext = os.path.splitext(trajectory)
        output = '{}.frame{}.inpcrd'.format(basename, nframe)

    # ParmEd sometimes struggles with certain PRMTOP files
    if os.path.splitext(topology)[1] in ('.top', '.prmtop'):
        top = AmberPrmtopFile(topology)
        mdtop = mdtraj.Topology.from_openmm(top.topology)
        traj = mdtraj.load_frame(trajectory, int(nframe), top=mdtop)
        structure = parmed.openmm.load_topology(top.topology, system=top.createSystem())
        structure.box_vectors = top.topology.getPeriodicBoxVectors()

    else:  # standard protocol (the topology is loaded twice, though)
        traj = mdtraj.load_frame(trajectory, int(nframe), top=topology)
        structure = parmed.load_file(topology)

    structure.positions = traj.openmm_positions(0)

    if traj.unitcell_vectors is not None:  # if frame provides box vectors, use those
        structure.box_vectors = traj.openmm_boxes(0)

    structure.save(output, overwrite=True)