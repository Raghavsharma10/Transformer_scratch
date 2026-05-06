def get_freesurfer_label(annot_input, verbose = True):
    """
    Print freesurfer label names.
    """
    labels, color_table, names = nib.freesurfer.read_annot(annot_input)
    if verbose:
        print(names)
    return names