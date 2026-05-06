def add_optional_parameters(detail_json, detail, rating, rating_n, popularity, current_popularity, time_spent):
    """
    check for optional return parameters and add them to the result json
    :param detail_json:
    :param detail:
    :param rating:
    :param rating_n:
    :param popularity:
    :param current_popularity:
    :param time_spent:
    :return:
    """

    if rating is not None:
        detail_json["rating"] = rating
    elif "rating" in detail:
        detail_json["rating"] = detail["rating"]

    if rating_n is not None:
        detail_json["rating_n"] = rating_n

    if "international_phone_number" in detail:
        detail_json["international_phone_number"] = detail["international_phone_number"]

    if current_popularity is not None:
        detail_json["current_popularity"] = current_popularity

    if popularity is not None:
        popularity, wait_times = get_popularity_for_day(popularity)

        detail_json["populartimes"] = popularity
        detail_json["time_wait"] = wait_times

    if time_spent is not None:
        detail_json["time_spent"] = time_spent

    return detail_json