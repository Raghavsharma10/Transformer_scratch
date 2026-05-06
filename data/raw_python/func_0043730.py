def position_result_list(change_list):
    """
    Returns a template which iters through the models and appends a new
    position column.

    """
    result = result_list(change_list)
    # Remove sortable attributes
    for x in range(0, len(result['result_headers'])):
        result['result_headers'][x]['sorted'] = False
        if result['result_headers'][x]['sortable']:
            result['result_headers'][x]['class_attrib'] = mark_safe(
                ' class="sortable"')
    # Append position <th> element
    result['result_headers'].append({
        'url_remove': '?o=',
        'sort_priority': 1,
        'sortable': True,
        'class_attrib': mark_safe(' class="sortable sorted ascending"'),
        'sorted': True,
        'text': 'position',
        'ascending': True,
        'url_primary': '?o=-1',
        'url_toggle': '?o=-1',
    })
    # Append the editable field to every result item
    for x in range(0, len(result['results'])):
        obj = change_list.result_list[x]
        # Get position object
        c_type = ContentType.objects.get_for_model(obj)
        try:
            object_position = ObjectPosition.objects.get(
                content_type__pk=c_type.id, object_id=obj.id)
        except ObjectPosition.DoesNotExist:
            object_position = ObjectPosition.objects.create(content_object=obj)
        # Add the <td>
        html = ('<td><input class="vTextField" id="id_position-{0}"'
                ' maxlength="10" name="position-{0}" type="text"'
                ' value="{1}" /></td>').format(object_position.id,
                                               object_position.position)
        result['results'][x].append(mark_safe(html))
    return result