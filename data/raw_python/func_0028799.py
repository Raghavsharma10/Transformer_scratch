def post_molo_comment(request, next=None, using=None):
    """
    Allows for posting of a Molo Comment, this allows comments to
    be set with the "user_name" as "Anonymous"
    """
    data = request.POST.copy()
    if 'submit_anonymously' in data:
        data['name'] = 'Anonymous'
    # replace with our changed POST data

    # ensure we always set an email
    data['email'] = request.user.email or 'blank@email.com'

    request.POST = data
    return post_comment(request, next=next, using=next)