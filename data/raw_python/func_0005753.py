def update_with_tohu_generators(field_gens, adict):
    """
    Helper function which updates `field_gens` with any items in the
    dictionary `adict` that are instances of `TohuUltraBaseGenerator`.
    """
    for name, gen in adict.items():
        if isinstance(gen, TohuUltraBaseGenerator):
            field_gens[name] = gen