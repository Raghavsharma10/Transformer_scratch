def _check(user, topic):
    """If the topic has it's export_control set to True then all the teams
    under the product team can access to the topic's resources.

    :param user:
    :param topic:
    :return: True if check is ok, False otherwise
    """
    # if export_control then check the team is associated to the product, ie.:
    #   - the current user belongs to the product's team
    #   OR
    #   - the product's team belongs to the user's parents teams
    if topic['export_control']:
        product = v1_utils.verify_existence_and_get(topic['product_id'],
                                                    models.PRODUCTS)
        return (user.is_in_team(product['team_id']) or
                product['team_id'] in user.parent_teams_ids)
    return False