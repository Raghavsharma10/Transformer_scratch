def compat_serializer_check_is_valid(serializer):
    """ http://www.django-rest-framework.org/topics/3.0-announcement/#using-is_validraise_exceptiontrue """
    if DRFVLIST[0] >= 3:
        serializer.is_valid(raise_exception=True)
    else:
        if not serializer.is_valid():
            serializers.ValidationError('The serializer raises a validation error')