def Voevent(stream, stream_id, role):
    """Create a new VOEvent element tree, with specified IVORN and role.

    Args:

        stream (str): used to construct the IVORN like so::

                ivorn = 'ivo://' + stream + '#' + stream_id

            (N.B. ``stream_id`` is converted to string if required.)
            So, e.g. we might set::

                stream='voevent.soton.ac.uk/super_exciting_events'
                stream_id=77

        stream_id (str): See above.

        role (str): role as defined in VOEvent spec.
            (See also  :py:class:`.definitions.roles`)

    Returns:
        Root-node of the VOEvent, as represented by an lxml.objectify element
        tree ('etree'). See also
        http://lxml.de/objectify.html#the-lxml-objectify-api
    """
    parser = objectify.makeparser(remove_blank_text=True)
    v = objectify.fromstring(voeventparse.definitions.v2_0_skeleton_str,
                             parser=parser)
    _remove_root_tag_prefix(v)
    if not isinstance(stream_id, string_types):
        stream_id = repr(stream_id)
    v.attrib['ivorn'] = ''.join(('ivo://', stream, '#', stream_id))
    v.attrib['role'] = role
    # Presumably we'll always want the following children:
    # (NB, valid to then leave them empty)
    etree.SubElement(v, 'Who')
    etree.SubElement(v, 'What')
    etree.SubElement(v, 'WhereWhen')
    v.Who.Description = (
        'VOEvent created with voevent-parse, version {}. '
        'See https://github.com/timstaley/voevent-parse for details.').format(
        __version__
    )
    return v