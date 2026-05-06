def fraction_digits(n: Literal) -> Optional[int]:
    """ 5.4.5 XML Schema Numeric Facet Constraints

    for "fractiondigits" constraints, v is less than or equals the number of digits to the right of the decimal place
    in the XML Schema canonical form[xmlschema-2] of the value of n, ignoring trailing zeros.
    """
    # Note - the last expression below isolates the fractional portion, reverses it (e.g. 017320 --> 023710) and
    #        converts it to an integer and back to a string
    return None if not is_numeric(n) or n.value is None \
        else 0 if is_integer(n) or '.' not in str(n.value) or str(n.value).split('.')[1] == '0' \
        else len(str(int(str(n.value).split('.')[1][::-1])))