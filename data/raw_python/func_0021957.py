def generate_random_missense_variants(num_variants=10, max_search=100000, reference="GRCh37"):
    """
    Generate a random collection of missense variants by trying random variants repeatedly.
    """
    variants = []
    for i in range(max_search):
        bases = ["A", "C", "T", "G"]
        random_ref = choice(bases)
        bases.remove(random_ref)
        random_alt = choice(bases)
        random_contig = choice(["1", "2", "3", "4", "5"])
        random_variant = Variant(contig=random_contig, start=randint(1, 1000000),
                                 ref=random_ref, alt=random_alt, ensembl=reference)
        try:
            effects = random_variant.effects()
            for effect in effects:
                if isinstance(effect, Substitution):
                    variants.append(random_variant)
                    break
        except:
            continue
        if len(variants) == num_variants:
            break
    return VariantCollection(variants)