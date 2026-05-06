def is_simple_literal(n: Node) -> bool:
    """ simple literal denotes a plain literal with no language tag. """
    return is_typed_literal(n) and cast(Literal, n).datatype is None and cast(Literal, n).language is None