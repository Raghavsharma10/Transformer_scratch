def is_strict_numeric(n: Node) -> bool:
    """ numeric denotes typed literals with datatypes xsd:integer, xsd:decimal, xsd:float, and xsd:double. """
    return is_typed_literal(n) and cast(Literal, n).datatype in [XSD.integer, XSD.decimal, XSD.float, XSD.double]