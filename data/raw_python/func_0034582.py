def post(request):
    """Returns a serialized object"""
    data = request.POST or json.loads(request.body)['body']
    guid = data.get('guid', None)
    res = Result()

    if guid:
        obj = getObjectsFromGuids([guid,])[0]
        comment = Comment()
        comment.comment = data.get('comment', 'No comment')
        comment.user = request.user
        comment.user_name = request.user.get_full_name()
        comment.user_email = request.user.email
        comment.content_object = obj
        # For our purposes, we never have more than one site
        comment.site_id = 1
        comment.save()

        obj.comment_count += 1
        obj.save()

        emailComment(comment, obj, request)

        res.append(commentToJson(comment))

    return JsonResponse(res.asDict())