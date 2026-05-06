def ancestors(obj, refattrs=(ALIGNMENT, SEGMENTATION)):
    """
    >>> for anc in query.ancestors(igt.get_item('g1'), refattrs=(ALIGNMENT, SEGMENTATION)):
    ...     print(anc)
    (<Tier object (id: g type: glosses) at ...>, 'alignment', <Tier object (id: m type: morphemes) at ...>, [<Item object (id: m1) at ...>])
    (<Tier object (id: m type: morphemes) at ...>, 'segmentation', <Tier object (id: w type: words) at ...>, [<Item object (id: w1) at ...>])
    (<Tier object (id: w type: words) at ...>, 'segmentation', <Tier object (id: p type: phrases) at ...>, [<Item object (id: p1) at ...>])
    """
    if hasattr(obj, 'tier'):
        tier = obj.tier
        items = [obj]
    else:
        tier = obj
        items = tier.items
    # a tier may be visited twice (e.g. A > B > A), but then it stops;
    # this is to avoid cycles
    visited = set([tier.id])
    while True:
        # get the first specified attribute
        refattr = next((ra for ra in refattrs if ra in tier.attributes), None)
        if not refattr:
            break
        reftier = ref.dereference(tier, refattr)
        ids = set(chain.from_iterable(
            ref.ids(item.attributes.get(refattr, '')) for item in items
        ))
        refitems = [item for item in reftier.items if item.id in ids]
        yield (tier, refattr, reftier, refitems)
        # cycle detection; break if we've now encountered something twice
        if reftier.id in visited:
            break
        visited.update(reftier.id)
        tier = reftier
        items = refitems