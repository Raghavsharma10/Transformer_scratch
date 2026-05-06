def negate_gate(wordlen, input='x', output='~x'):
    """Implements two's complement negation."""
    neg = bitwise_negate(wordlen, input, "tmp")
    inc = inc_gate(wordlen, "tmp", output)
    return neg >> inc