def filter_effects(effect_collection, variant_collection, patient, filter_fn, all_effects, **kwargs):
    """Filter variants from the Effect Collection

    Parameters
    ----------
    effect_collection : varcode.EffectCollection
    variant_collection : varcode.VariantCollection
    patient : cohorts.Patient
    filter_fn : function
        Takes a FilterableEffect and returns a boolean. Only effects returning True are preserved.
    all_effects : boolean
        Return the single, top-priority effect if False. If True, return all effects (don't filter to top-priority).

    Returns
    -------
    varcode.EffectCollection
        Filtered effect collection, with only the variants passing the filter
    """
    def top_priority_maybe(effect_collection):
        """
        Always (unless all_effects=True) take the top priority effect per variant
        so we end up with a single effect per variant.
        """
        if all_effects:
            return effect_collection
        return EffectCollection(list(effect_collection.top_priority_effect_per_variant().values()))

    def apply_filter_fn(filter_fn, effect):
        """
        Return True if filter_fn is true for the effect or its alternate_effect.
        If no alternate_effect, then just return True if filter_fn is True.
        """
        applied = filter_fn(FilterableEffect(
            effect=effect,
            variant_collection=variant_collection,
            patient=patient), **kwargs)
        if hasattr(effect, "alternate_effect"):
            applied_alternate = filter_fn(FilterableEffect(
                effect=effect.alternate_effect,
                variant_collection=variant_collection,
                patient=patient), **kwargs)
            return applied or applied_alternate
        return applied

    if filter_fn:
        return top_priority_maybe(EffectCollection([
            effect
            for effect in effect_collection
            if apply_filter_fn(filter_fn, effect)]))
    else:
        return top_priority_maybe(effect_collection)