def api_ebuio_forum(request, key=None, hproPk=None):
    """Create a topic on the forum of the ioproject. EBUIo only !"""

    if not check_api_key(request, key, hproPk):
        return HttpResponseForbidden

    if settings.PIAPI_STANDALONE:
        return HttpResponse(json.dumps({'error': 'no-on-ebuio'}), content_type="application/json")

    (_, _, hproject) = getPlugItObject(hproPk)

    error = ''

    subject = request.POST.get('subject')
    author_pk = request.POST.get('author')
    message = request.POST.get('message')
    tags = request.POST.get('tags', '')

    if not subject:
        error = 'no-subject'
    if not author_pk:
        error = 'no-author'
    else:
        try:
            from users.models import TechUser

            author = TechUser.objects.get(pk=author_pk)
        except TechUser.DoesNotExist:
            error = 'author-no-found'

    if not message:
        error = 'no-message'

    if error:
        return HttpResponse(json.dumps({'error': error}), content_type="application/json")

    # Create the topic
    from discuss.models import Post, PostTag

    if tags:
        real_tags = []
        for tag in tags.split(','):
            (pt, __) = PostTag.objects.get_or_create(tag=tag)
            real_tags.append(str(pt.pk))

        tags = ','.join(real_tags)

    post = Post(content_object=hproject, who=author, score=0, title=subject, text=message)
    post.save()

    from app.tags_utils import update_object_tag

    update_object_tag(post, PostTag, tags)

    post.send_email()

    # Return the URL
    return HttpResponse(json.dumps({'result': 'ok',
                                    'url': settings.EBUIO_BASE_URL + reverse('hprojects.views.forum_topic',
                                                                             args=(hproject.pk, post.pk))}),
                        content_type="application/json")