def get(request, obj_id):
    """Returns a serialized object
    :param obj_id: ID of comment object
    :type obj_id: int
    :returns: json
    """
    res = Result()
    c = Comment.objects.get(pk=obj_id)
    res.append(commentToJson(c))

    return JsonResponse(res.asDict())