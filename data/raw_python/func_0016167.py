def make_avro_schema(i,         # type: List[Any]
                     loader     # type: Loader
                     ):  # type: (...) -> Names
    """
    All in one convenience function.

    Call make_avro() and make_avro_schema_from_avro() separately if you need
    the intermediate result for diagnostic output.
    """
    names = Names()
    avro = make_avro(i, loader)
    make_avsc_object(convert_to_dict(avro), names)
    return names