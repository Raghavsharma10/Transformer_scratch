def delete_secnos_factory(f):
    """Returns delete_secnos(key, value, fmt, meta) action that deletes
    section numbers from the attributes of elements of type f.
    """

    # Get the name and standard length
    name = f.__closure__[0].cell_contents
    n = f.__closure__[1].cell_contents

    def delete_secnos(key, value, fmt, meta):  # pylint: disable=unused-argument
        """Deletes section numbers from elements attributes."""
        if 'xnos-number-sections' in meta and \
          check_bool(get_meta(meta, 'xnos-number-sections')) and \
              fmt in ['html', 'html5']:

            # Only delete if attributes are attached.   Images always have
            # attributes.
            if key == name:
                assert len(value) <= n+1
                if name == 'Image' or len(value) == n+1:

                    # Make sure value[0] represents attributes
                    assert len(value[0]) == 3
                    assert isinstance(value[0][0], STRTYPES)
                    assert isinstance(value[0][1], list)
                    assert isinstance(value[0][2], list)

                    # Remove the secno attribute
                    if value[0][2] and value[0][2][0][0] == 'secno':
                        del value[0][2][0]

    return delete_secnos