def render_to_mail(template, context, **kwargs):
    """
    Renders a mail and returns the resulting ``EmailMultiAlternatives``
    instance

    * ``template``: The base name of the text and HTML (optional) version of
      the mail.
    * ``context``: The context used to render the mail. This context instance
      should contain everything required.
    * Additional keyword arguments are passed to the ``EmailMultiAlternatives``
      instantiation. Use those to specify the ``to``, ``headers`` etc.
      arguments.

    Usage example::

        # Render the template myproject/hello_mail.txt (first non-empty line
        # contains the subject, third to last the body) and optionally the
        # template myproject/hello_mail.html containing the alternative HTML
        # representation.
        message = render_to_mail('myproject/hello_mail', {}, to=[email])
        message.send()
    """
    lines = iter(
        line.rstrip()
        for line in render_to_string("%s.txt" % template, context).splitlines()
    )

    subject = ""
    try:
        while True:
            line = next(lines)
            if line:
                subject = line
                break
    except StopIteration:  # if lines is empty
        pass

    body = "\n".join(lines).strip("\n")
    message = EmailMultiAlternatives(subject=subject, body=body, **kwargs)

    try:
        message.attach_alternative(
            render_to_string("%s.html" % template, context), "text/html"
        )
    except TemplateDoesNotExist:
        pass

    return message