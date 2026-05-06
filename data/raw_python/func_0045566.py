def email_template(rcpt, template_path, **kwargs):
    """ Load, render and email template.
        **kwargs may contain variables for template rendering.
    """

    subject, content = parse_template(template_path, **kwargs)
    count = send_mail(subject, content, settings.DEFAULT_FROM_EMAIL,
                      [rcpt], fail_silently=True)
    return bool(count)