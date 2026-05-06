def to_email(email_class, email, language=None, **data):
    """
    Send email to specified email address
    """
    if language:
        email_class().send([email], language=language, **data)
    else:
        email_class().send([email], translation.get_language(), **data)