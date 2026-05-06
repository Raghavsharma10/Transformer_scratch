def compat_get_paginated_response(view, page):
    """ get_paginated_response is unknown to DRF 3.0 """
    if DRFVLIST[0] == 3 and DRFVLIST[1] >= 1:
        from rest_messaging.serializers import ComplexMessageSerializer  # circular import
        serializer = ComplexMessageSerializer(page, many=True)
        return view.get_paginated_response(serializer.data)
    else:
        serializer = view.get_pagination_serializer(page)
        return Response(serializer.data)