def can_cast_to(v: Literal, dt: str) -> bool:
    """ 5.4.3 Datatype Constraints

    Determine whether "a value of the lexical form of n can be cast to the target type v per
    XPath Functions 3.1 section 19 Casting[xpath-functions]."
    """
    # TODO: rdflib doesn't appear to pay any attention to lengths (e.g. 257 is a valid XSD.byte)
    return v.value is not None and Literal(str(v), datatype=dt).value is not None