def enforce_relationship_refs(instance):
    """Ensures that all SDOs being referenced by the SRO are contained
    within the same bundle"""
    if instance['type'] != 'bundle' or 'objects' not in instance:
        return

    rel_references = set()

    """Find and store all ids"""
    for obj in instance['objects']:
        if obj['type'] != 'relationship':
            rel_references.add(obj['id'])

    """Check if id has been encountered"""
    for obj in instance['objects']:
        if obj['type'] == 'relationship':
            if obj['source_ref'] not in rel_references:
                yield JSONError("Relationship object %s makes reference to %s "
                                "Which is not found in current bundle "
                                % (obj['id'], obj['source_ref']), 'enforce-relationship-refs')

            if obj['target_ref'] not in rel_references:
                yield JSONError("Relationship object %s makes reference to %s "
                                "Which is not found in current bundle "
                                % (obj['id'], obj['target_ref']), 'enforce-relationship-refs')