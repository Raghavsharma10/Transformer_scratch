def group_data():
    """ Load the reference data, and assign each object
    a random integer from 0 to 7. Save the IDs. """

    tr_obj = np.load("%s/ref_id.npz" %direc_ref)['arr_0']
    groups = np.random.randint(0, 8, size=len(tr_obj))
    np.savez("ref_groups.npz", groups)