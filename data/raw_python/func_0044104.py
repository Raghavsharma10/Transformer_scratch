def load_pdb(pdb, path=True, pdb_id='', ignore_end=False):
    """Converts a PDB file into an AMPAL object.

    Parameters
    ----------
    pdb : str
        Either a path to a PDB file or a string containing PDB
        format structural data.
    path : bool, optional
        If `true`, flags `pdb` as a path and not a PDB string.
    pdb_id : str, optional
        Identifier for the `Assembly`.
    ignore_end : bool, optional
        If `false`, parsing of the file will stop when an "END"
        record is encountered.

    Returns
    -------
    ampal : ampal.Assembly or ampal.AmpalContainer
        AMPAL object that contains the structural information from
        the PDB file provided. If the PDB file has a single state
        then an `Assembly` will be returned, otherwise an
        `AmpalContainer` will be returned.
    """
    pdb_p = PdbParser(pdb, path=path, pdb_id=pdb_id, ignore_end=ignore_end)
    return pdb_p.make_ampal()