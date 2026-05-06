def decode(code, *, max_age):
    """decode(code, *, max_age)
    Decodes the code from the registration link and returns a tuple consisting
    of the verified email address and the payload which was passed through to
    ``get_confirmation_code``.

    The maximum age in seconds of the link has to be specified as ``max_age``.

    This method raises ``ValidationError`` exceptions when anything goes wrong
    when verifying the signature or the expiry timeout.
    """
    try:
        data = get_signer().unsign(code, max_age=max_age)
    except signing.SignatureExpired:
        raise ValidationError(
            _("The link is expired. Please request another registration link."),
            code="email_registration_expired",
        )

    except signing.BadSignature:
        raise ValidationError(
            _(
                "Unable to verify the signature. Please request a new"
                " registration link."
            ),
            code="email_registration_signature",
        )

    return data.split(":", 1)