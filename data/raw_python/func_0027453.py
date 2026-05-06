def verify_team_in_topic(user, topic_id):
    """Verify that the user's team does belongs to the given topic. If
    the user is an admin or read only user then it belongs to all topics.
    """
    if user.is_super_admin() or user.is_read_only_user():
        return
    if str(topic_id) not in user_topic_ids(user):
        raise dci_exc.Unauthorized()