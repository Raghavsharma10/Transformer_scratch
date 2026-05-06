def update_parent_sequence_map(child_part, delete=False):
    """Updates the child map of a simple sequence assessment assessment part"""
    if child_part.has_parent_part():
        object_map = child_part.get_assessment_part()._my_map
        database = 'assessment_authoring'
        collection_type = 'AssessmentPart'
    else:
        object_map = child_part.get_assessment()._my_map
        database = 'assessment'
        collection_type = 'Assessment'
    collection = JSONClientValidated(database,
                                     collection=collection_type,
                                     runtime=child_part._runtime)
    if delete and 'childIds' in object_map:
        object_map['childIds'].remove(str(child_part.get_id()))
    elif not delete:
        if 'childIds' not in object_map:
            object_map['childIds'] = []
        object_map['childIds'].append(str(child_part.get_id()))
    collection.save(object_map)