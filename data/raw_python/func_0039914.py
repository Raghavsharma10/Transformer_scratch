def render_email_template(email_template, base_url, extra_context=None, user=None):
    """
    Render the email template.

    :type email_template: fluentcms_emailtemplates.models.EmailTemplate
    :type base_url: str
    :type extra_context: dict | None
    :type user: django.contrib.auth.models.User
    :return: The subject, html and text content
    :rtype: fluentcms_emailtemplates.rendering.EmailContent
    """
    dummy_request = _get_dummy_request(base_url, user)
    context_user = user or extra_context.get('user', None)

    context_data = {
        'request': dummy_request,
        'email_template': email_template,
        'email_format': 'html',
        'user': user,
        # Common replacements
        'first_name': context_user.first_name if context_user else '',
        'last_name': context_user.last_name if context_user else '',
        'full_name': context_user.get_full_name() if context_user else '',
        'email': context_user.email if context_user else '',
        'site': extra_context.get('site', None) or {
            'domain': dummy_request.get_host(),
            'name': dummy_request.get_host(),
        }
    }
    if extra_context:
        context_data.update(extra_context)

    # Make sure the templates and i18n are identical to the emailtemplate language.
    # This is the same as the current Django language, unless the object was explicitly fetched in a different language.
    with switch_language(email_template):
        # Get the body content
        context_data['body'] = _render_email_placeholder(dummy_request, email_template, base_url, context_data)
        context_data['subject'] = subject = replace_fields(email_template.subject, context_data, autoescape=False)

        # Merge that with the HTML templates.
        context = RequestContext(dummy_request).flatten()
        context.update(context_data)
        html = render_to_string(email_template.get_html_templates(), context, request=dummy_request)
        html, url_changes = _make_links_absolute(html, base_url)

        # Render the Text template.
        # Disable auto escaping
        context['email_format'] = 'text'
        text = render_to_string(email_template.get_text_templates(), context, request=dummy_request)
        text = _make_text_links_absolute(text, url_changes)

        return EmailContent(subject, text, html)