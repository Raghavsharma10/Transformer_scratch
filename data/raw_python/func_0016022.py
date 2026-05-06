def observable_object_keys(instance):
    """Ensure observable-objects keys are non-negative integers.
    """
    digits_re = re.compile(r"^\d+$")
    for key in instance['objects']:
        if not digits_re.match(key):
            yield JSONError("'%s' is not a good key value. Observable Objects "
                            "should use non-negative integers for their keys."
                            % key, instance['id'], 'observable-object-keys')