def remove_duplicates(apps, schema_editor):
    """
    Remove any duplicates from the entity relationship table
    :param apps:
    :param schema_editor:
    :return:
    """

    # Get the model
    EntityRelationship = apps.get_model('entity', 'EntityRelationship')

    # Find the duplicates
    duplicates = EntityRelationship.objects.all().order_by(
        'sub_entity_id',
        'super_entity_id'
    ).values(
        'sub_entity_id',
        'super_entity_id'
    ).annotate(
        Count('sub_entity_id'),
        Count('super_entity_id'),
        max_id=Max('id')
    ).filter(
        super_entity_id__count__gt=1
    )

    # Loop over the duplicates and delete
    for duplicate in duplicates:
        EntityRelationship.objects.filter(
            sub_entity_id=duplicate['sub_entity_id'],
            super_entity_id=duplicate['super_entity_id']
        ).exclude(
            id=duplicate['max_id']
        ).delete()