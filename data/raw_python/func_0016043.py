def granular_markings_circular_refs(instance):
    """Ensure that marking definitions do not contain circular references (ie.
    they do not reference themselves in the `granular_markings` property).
    """
    if instance['type'] != 'marking-definition':
        return

    if 'granular_markings' in instance:
        for marking in instance['granular_markings']:
            if 'marking_ref' in marking and marking['marking_ref'] == instance['id']:
                yield JSONError("`granular_markings` cannot contain any "
                                "references to this marking definition object"
                                " (no circular references).", instance['id'])