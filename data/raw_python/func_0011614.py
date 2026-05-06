def phonenumber_validation(data):
    """ Validates phonenumber

    Similar to phonenumber_field.validators.validate_international_phonenumber() but uses a different message if the
    country prefix is absent.
    """
    from phonenumber_field.phonenumber import to_python
    phone_number = to_python(data)
    if not phone_number:
        return data
    elif not phone_number.country_code:
        raise serializers.ValidationError(_("Phone number needs to include valid country code (E.g +37255555555)."))
    elif not phone_number.is_valid():
        raise serializers.ValidationError(_('The phone number entered is not valid.'))

    return data