def _get_content_type_queryset(models_list):
    """ Get list of services content types """
    content_type_ids = {c.id for c in ContentType.objects.get_for_models(*models_list).values()}
    return ContentType.objects.filter(id__in=content_type_ids)