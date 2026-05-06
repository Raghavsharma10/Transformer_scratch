def remove_from_parent_sequence_map(assessment_part_admin_session, assessment_part_id):
    """Updates the child map of a simple sequence assessment assessment part to remove child part"""
    apls = get_assessment_part_lookup_session(runtime=assessment_part_admin_session._runtime,
                                              proxy=assessment_part_admin_session._proxy)
    apls.use_federated_bank_view()
    apls.use_unsequestered_assessment_part_view()
    child_part = apls.get_assessment_part(assessment_part_id)
    update_parent_sequence_map(child_part, delete=True)