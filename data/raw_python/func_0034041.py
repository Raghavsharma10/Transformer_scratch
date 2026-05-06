def api_ebuio_forum_get_topics_by_tag_for_user(request, key=None, hproPk=None, tag=None, userPk=None):
    """Return the list of topics using the tag pk"""

    # Check API key (in order to be sure that we have a valid one and that's correspond to the project
    if not check_api_key(request, key, hproPk):
        return HttpResponseForbidden

    if settings.PIAPI_STANDALONE:
        return HttpResponse(json.dumps({'error': 'no-on-ebuio'}), content_type="application/json")

    # We get the plugit object representing the project
    (_, _, hproject) = getPlugItObject(hproPk)

    # We get the user and we check his rights
    author_pk = request.GET.get('u')
    if author_pk and author_pk.isdigit():
        try:
            from users.models import TechUser

            user = TechUser.objects.get(pk=author_pk)
        except TechUser.DoesNotExist:
            error = 'user-no-found'
            user = generate_user(mode='ano')
    else:
        user = generate_user(mode='ano')

    if not hproject.discuss_can_display_posts(user):
        return HttpResponseForbidden

    # Verify the existence of the tag
    if not tag:
        raise Http404

    # We get the posts (only topics ones-the parent) related to the project and to the tag.
    # We dont' take the deleted ones.
    from discuss.models import Post

    posts = Post.objects.filter(is_deleted=False).filter(object_id=hproPk).filter(tags__tag=tag).order_by('-when')

    # We convert the posts list to json
    posts_json = [
        {'id': post.id, 'link': post.discuss_get_forum_topic_link(), 'subject': post.title, 'author': post.who_id,
         'when': post.when.strftime('%a, %d %b %Y %H:%M GMT'), 'score': post.score,
         'replies_number': post.direct_subposts_size()} for post in posts]

    return HttpResponse(json.dumps({'data': posts_json}), content_type="application/json")