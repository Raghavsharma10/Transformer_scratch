def emailUser(video, error=None):
    """Emails the author of the video that it has finished processing"""
    html = render_to_string('frog/video_email.html', {
        'user': video.author,
        'error': error,
        'video': video,
        'SITE_URL': FROG_SITE_URL,
    })
    subject, from_email, to = 'Video Processing Finished{}'.format(error or ''), 'noreply@frogmediaserver.com', video.author.email
    text_content = 'This is an important message.'
    html_content = html

    send_mail(subject, text_content, from_email, [to], html_message=html_content)