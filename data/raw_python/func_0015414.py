def nodeSatisfiesNumericFacet(cntxt: Context, n: Node, nc: ShExJ.NodeConstraint, _c: DebugContext) -> bool:
    """ `5.4.5 XML Schema Numeric Facet Constraints <http://shex.io/shex-semantics/#xs-numeric>`_

    Numeric facet constraints apply to the numeric value of RDF Literals with datatypes listed in SPARQL 1.1
    Operand Data Types[sparql11-query]. Numeric constraints on non-numeric values fail. totaldigits and
    fractiondigits constraints on values not derived from xsd:decimal fail.
    """
    if nc.mininclusive is not None or nc.minexclusive is not None or nc.maxinclusive is not None \
            or nc.maxexclusive is not None or nc.totaldigits is not None or nc.fractiondigits is not None:
        if is_numeric(n):
            v = n.value
            if isinstance(v, numbers.Number):
                if (nc.mininclusive is None or v >= nc.mininclusive) and \
                   (nc.minexclusive is None or v > nc.minexclusive) and \
                   (nc.maxinclusive is None or v <= nc.maxinclusive) and \
                   (nc.maxexclusive is None or v < nc.maxexclusive) and \
                   (nc.totaldigits is None or (total_digits(n) is not None and
                                                   total_digits(n) <= nc.totaldigits)) and \
                   (nc.fractiondigits is None or (fraction_digits(n) is not None and
                                                      fraction_digits(n) <= nc.fractiondigits)):
                    return True
                else:
                    if nc.mininclusive is not None and v < nc.mininclusive:
                        cntxt.fail_reason = f"Numeric value volation - minimum inclusive: " \
                                                         f"{nc.mininclusive} actual: {v}"
                    elif nc.minexclusive is not None and v <= nc.minexclusive:
                        cntxt.fail_reason = f"Numeric value volation - minimum exclusive: " \
                                                         f"{nc.minexclusive} actual: {v}"
                    elif nc.maxinclusive is not None and v > nc.maxinclusive:
                        cntxt.fail_reason = f"Numeric value volation - maximum inclusive: " \
                                                         f"{nc.maxinclusive} actual: {v}"
                    elif nc.maxexclusive is not None and v >= nc.maxexclusive:
                        cntxt.fail_reason = f"Numeric value volation - maximum exclusive: " \
                                                         f"{nc.maxexclusive} actual: {v}"
                    elif nc.totaldigits is not None and (total_digits(n) is None or
                                                             total_digits(n) > nc.totaldigits):
                        cntxt.fail_reason = f"Numeric value volation - max total digits: " \
                                                         f"{nc.totaldigits} value: {v}"
                    elif nc.fractiondigits is not None and (fraction_digits(n) is None or
                                                                total_digits(n) > nc.fractiondigits):
                        cntxt.fail_reason = f"Numeric value volation - max fractional digits: " \
                                                         f"{nc.fractiondigits} value: {v}"
                    else:
                        cntxt.fail_reason = "Impossible error - kick the programmer"
                    return False
            else:
                cntxt.fail_reason = "Numeric test on non-number: {v}"
                return False
        else:
            cntxt.fail_reason = "Numeric test on non-number: {n}"
            return False
    return True