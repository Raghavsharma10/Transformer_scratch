def replace_type(items, spec, loader, found, find_embeds=True, deepen=True):
    # type: (Any, Dict[Text, Any], Loader, Set[Text], bool, bool) -> Any
    """ Go through and replace types in the 'spec' mapping"""

    if isinstance(items, MutableMapping):
        # recursively check these fields for types to replace
        if items.get("type") in ("record", "enum") and items.get("name"):
            if items["name"] in found:
                return items["name"]
            found.add(items["name"])

        if not deepen:
            return items

        items = copy.copy(items)
        if not items.get("name"):
            items["name"] = get_anon_name(items)
        for name in ("type", "items", "fields"):
            if name in items:
                items[name] = replace_type(
                    items[name], spec, loader, found, find_embeds=find_embeds,
                    deepen=find_embeds)
                if isinstance(items[name], MutableSequence):
                    items[name] = flatten(items[name])

        return items
    if isinstance(items, MutableSequence):
        # recursively transform list
        return [replace_type(i, spec, loader, found, find_embeds=find_embeds,
                             deepen=deepen) for i in items]
    if isinstance(items, string_types):
        # found a string which is a symbol corresponding to a type.
        replace_with = None
        if items in loader.vocab:
            # If it's a vocabulary term, first expand it to its fully qualified
            # URI
            items = loader.vocab[items]

        if items in spec:
            # Look up in specialization map
            replace_with = spec[items]

        if replace_with:
            return replace_type(replace_with, spec, loader, found, find_embeds=find_embeds)
        found.add(items)
    return items