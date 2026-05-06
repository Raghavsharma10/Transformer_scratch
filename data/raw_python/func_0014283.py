def from_parmed(cls, path, *args, **kwargs):
        """
        Try to load a file automatically with ParmEd. Not guaranteed to work, but
        might be useful if it succeeds.

        Arguments
        ---------
        path : str
            Path to file that ParmEd can load
        """
        st = parmed.load_file(path, structure=True, *args, **kwargs)
        box = kwargs.pop('box', getattr(st, 'box', None))
        velocities = kwargs.pop('velocities', getattr(st, 'velocities', None))
        positions = kwargs.pop('positions', getattr(st, 'positions', None))
        return cls(master=st, topology=st.topology, positions=positions, box=box,
                   velocities=velocities, path=path, **kwargs)