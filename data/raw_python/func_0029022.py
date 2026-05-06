def send_registration_mail(email, *, request, **kwargs):
    """send_registration_mail(email, *, request, **kwargs)
    Sends the registration mail

    * ``email``: The email address where the registration link should be
      sent to.
    * ``request``: A HTTP request instance, used to construct the complete
      URL (including protocol and domain) for the registration link.
    * Additional keyword arguments for ``get_confirmation_url`` respectively
      ``get_confirmation_code``.

    The mail is rendered using the following two templates:

    * ``registration/email_registration_email.txt``: The first line of this
      template will be the subject, the third to the last line the body of the
      email.
    * ``registration/email_registration_email.html``: The body of the HTML
      version of the mail. This template is **NOT** available by default and
      is not required either.
    """

    render_to_mail(
        "registration/email_registration_email",
        {"url": get_confirmation_url(email, request, **kwargs)},
        to=[email],
    ).send()