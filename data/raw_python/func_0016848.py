def _apply_source_provider_dim_updates(cube, source_providers, budget_dims):
    """
    Given a list of source_providers, apply the list of
    suggested dimension updates given in provider.updated_dimensions()
    to the supplied hypercube.

    Dimension global_sizes are always updated with the supplied sizes and
    lower_extent is always set to 0. upper_extent is set to any reductions
    (current upper_extents) existing in budget_dims, otherwise it is set
    to global_size.

    """
    # Create a mapping between a dimension and a
    # list of (global_size, provider_name) tuples
    update_map = collections.defaultdict(list)

    for prov in source_providers:
        for dim_tuple in prov.updated_dimensions():
            name, size = dim_tuple

            # Don't accept any updates on the nsrc dimension
            # This is managed internally
            if name == 'nsrc':
                continue

            dim_update = DimensionUpdate(size, prov.name())
            update_map[name].append(dim_update)

    # No dimensions were updated, quit early
    if len(update_map) == 0:
        return cube.bytes_required()

    # Ensure that the global sizes we receive
    # for each dimension are unique. Tell the user
    # when conflicts occur
    update_list = []

    for name, updates in update_map.iteritems():
        if not all(updates[0].size == du.size for du in updates[1:]):
            raise ValueError("Received conflicting "
                "global size updates '{u}'"
                " for dimension '{n}'.".format(n=name, u=updates))

        update_list.append((name, updates[0].size))

    montblanc.log.info("Updating dimensions {} from "
                        "source providers.".format(str(update_list)))

    # Now update our dimensions
    for name, global_size in update_list:
        # Defer to existing any existing budgeted extent sizes
        # Otherwise take the global_size
        extent_size = budget_dims.get(name, global_size)

        # Take the global_size if extent_size was previously zero!
        extent_size = global_size if extent_size == 0 else extent_size

        # Clamp extent size to global size
        if extent_size > global_size:
            extent_size = global_size

        # Update the dimension
        cube.update_dimension(name,
            global_size=global_size,
            lower_extent=0,
            upper_extent=extent_size)

    # Handle global number of sources differently
    # It's equal to the number of
    # point's, gaussian's, sersic's combined
    nsrc = sum(cube.dim_global_size(*mbu.source_nr_vars()))

    # Extent size will be equal to whatever source type
    # we're currently iterating over. So just take
    # the maximum extent size given the sources
    es = max(cube.dim_extent_size(*mbu.source_nr_vars()))

    cube.update_dimension('nsrc',
        global_size=nsrc,
        lower_extent=0,
        upper_extent=es)

    # Return our cube size
    return cube.bytes_required()