def combine_and_save(add_path_list, out_path):
    """
    Combines whatever datasets that can be combined,
    and save the bigger dataset to a given location.
    """

    add_path_list = list(add_path_list)
    # first one!
    first_ds_path = add_path_list[0]
    print('Starting with {}'.format(first_ds_path))
    combined = MLDataset(first_ds_path)
    for ds_path in add_path_list[1:]:
        try:
            combined = combined + MLDataset(ds_path)
        except:
            print('      Failed to add {}'.format(ds_path))
            traceback.print_exc()
        else:
            print('Successfully added {}'.format(ds_path))

    combined.save(out_path)

    return