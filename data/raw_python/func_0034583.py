def put(request, obj_id):
    """Updates the content of a comment
    :param obj_id: ID of comment object
    :type obj_id: int
    :returns: json
    """
    res = Result()
    c = Comment.objects.get(pk=obj_id)
    data = request.PUT or json.loads(request.body)['body']
    content = data.get('comment', None)
    if content:
        c.comment = content
        c.save()

        res.append(commentToJson(c))

    return JsonResponse(res.asDict())