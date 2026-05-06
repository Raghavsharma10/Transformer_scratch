def insert_secnos_factory(f):
    """Returns insert_secnos(key, value, fmt, meta) action that inserts
    section numbers into the attributes of elements of type f.
    """

    # Get the name and standard length
    name = f.__closure__[0].cell_contents
    n = f.__closure__[1].cell_contents

    def insert_secnos(key, value, fmt, meta):  # pylint: disable=unused-argument
        """Inserts section numbers into elements attributes."""

        global sec  # pylint: disable=global-statement

        if 'xnos-number-sections' in meta and \
          check_bool(get_meta(meta, 'xnos-number-sections')) and \
              fmt in ['html', 'html5']:
            if key == 'Header':
                if 'unnumbered' in value[1][1]:
                    return
                level = value[0]
                m = level - len(sec)
                if m > 0:
                    sec.extend([0]*m)
                sec[level-1] += 1
                sec = sec[:MAXLEVEL]
            if key == name:

                # Only insert if attributes are attached.  Images always have
                # attributes.
                assert len(value) <= n+1
                if name == 'Image' or len(value) == n+1:

                    # Make sure value[0] represents attributes
                    assert len(value[0]) == 3
                    assert isinstance(value[0][0], STRTYPES)
                    assert isinstance(value[0][1], list)
                    assert isinstance(value[0][2], list)

                    # Insert the section number into the attributes
                    s = '.'.join([str(m) for m in sec])
                    value[0][2].insert(0, ['secno', s])

    return insert_secnos