def emailComment(comment, obj, request):
    """Send an email to the author about a new comment"""
    if not obj.author.frog_prefs.get().json()['emailComments']:
        return

    if obj.author == request.user:
        return

    html = render_to_string('frog/comment_email.html', {
        'user': comment.user,
        'comment': comment.comment,
        'object': obj,
        'action_type': 'commented on',
        'image': isinstance(obj, Image),
        'SITE_URL': FROG_SITE_URL,
    })

    subject = '{}: Comment from {}'.format(getSiteConfig()['name'], comment.user_name)
    fromemail = comment.user_email
    to = obj.author.email
    text_content = 'This is an important message.'
    html_content = html

    send_mail(subject, text_content, fromemail, [to], html_message=html_content)