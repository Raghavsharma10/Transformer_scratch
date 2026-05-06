def load_synapses(path=HOME + "/Downloads/pinky100_final.df",
                  scaling=(1, 1, 1)):
    """ Test scenario using real synapses """

    scaling = np.array(list(scaling))

    df = pd.read_csv(path)

    locs = np.array(df[["presyn_x", "centroid_x", "postsyn_x"]])

    mask = ~np.any(np.isnan(locs), axis=1)

    df = df[mask]

    df['pre_pt.position'] = list((np.array(df[['presyn_x', 'presyn_y', 'presyn_z']]) / scaling).astype(np.int))
    df['ctr_pt.position'] = list((np.array(df[['centroid_x', 'centroid_y', 'centroid_z']]) / scaling).astype(np.int))
    df['post_pt.position'] = list((np.array(df[['postsyn_x', 'postsyn_y', 'postsyn_z']]) / scaling).astype(np.int))

    df = df[['pre_pt.position', 'ctr_pt.position', 'post_pt.position', 'size']]

    return df