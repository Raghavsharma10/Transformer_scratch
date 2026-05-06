def Param(name, value=None, unit=None, ucd=None, dataType=None, utype=None,
          ac=True):
    """
    'Parameter', used as a general purpose key-value entry in the 'What' section.

    May be assembled into a :class:`Group`.

    NB ``name`` is not mandated by schema, but *is* mandated in full spec.

    Args:
        value(str): String representing parameter value.
            Or, if ``ac`` is true, then 'autoconversion' is attempted, in which case
            ``value`` can also be an instance of one of the following:

             * :py:obj:`bool`
             * :py:obj:`int`
             * :py:obj:`float`
             * :py:class:`datetime.datetime`

            This allows you to create Params without littering your code
            with string casts, or worrying if the passed value is a float or a
            string, etc.
            NB the value is always *stored* as a string representation,
            as per VO spec.
        unit(str): Units of value. See :class:`.definitions.units`
        ucd(str): `unified content descriptor <http://arxiv.org/abs/1110.0525>`_.
            For a list of valid UCDs, see:
            http://vocabularies.referata.com/wiki/Category:IVOA_UCD.
        dataType(str): Denotes type of ``value``; restricted to 3 options:
            ``string`` (default), ``int`` , or ``float``.
            (NB *not* to be confused with standard XML Datatypes, which have many
            more possible values.)
        utype(str): See http://wiki.ivoa.net/twiki/bin/view/IVOA/Utypes
        ac(bool): Attempt automatic conversion of passed ``value`` to string,
            and set ``dataType`` accordingly (only attempted if ``dataType``
            is the default, i.e. ``None``).
            (NB only supports types listed in _datatypes_autoconversion dict)

    """
    # We use locals() to allow concise looping over the arguments.
    atts = locals()
    atts.pop('ac')
    temp_dict = {}
    temp_dict.update(atts)
    for k in temp_dict.keys():
        if atts[k] is None:
            del atts[k]
    if (ac
        and value is not None
        and (not isinstance(value, string_types))
        and dataType is None
        ):
        if type(value) in _datatypes_autoconversion:
            datatype, func = _datatypes_autoconversion[type(value)]
            atts['dataType'] = datatype
            atts['value'] = func(value)
    return objectify.Element('Param', attrib=atts)