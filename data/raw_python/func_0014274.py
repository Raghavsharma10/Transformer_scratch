def process_forcefield(*forcefields):
    """
    Given a list of filenames, check which ones are `frcmods`. If so,
    convert them to ffxml. Else, just return them.
    """
    for forcefield in forcefields:
        if forcefield.endswith('.frcmod'):
            gaffmol2 = os.path.splitext(forcefield)[0] + '.gaff.mol2'
            yield create_ffxml_file([gaffmol2], [forcefield])
        else:
            yield forcefield