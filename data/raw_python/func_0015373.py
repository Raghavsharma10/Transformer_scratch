def total_digits(n: Literal) -> Optional[int]:
    """ 5.4.5 XML Schema Numberic Facet Constraints

     totaldigits and fractiondigits constraints on values not derived from xsd:decimal fail.
     """
    return len(str(abs(int(n.value)))) + fraction_digits(n) if is_numeric(n) and n.value is not None else None