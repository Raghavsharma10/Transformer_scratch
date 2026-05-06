def get_place_details(api_key, place_id, **kwargs):
    """
    sends request to detail to get a search string and uses standard proto buffer to get additional information
    on the current status of popular times
    :return: json details
    """

    params = {
        'placeid': place_id,
        'key': api_key,
        }

    resp = requests.get(url=DETAIL_GOOGLEMAPS_API_URL, params=params)

    if resp.status_code >= 300:
        raise Exception('Bad status code rerieved from google api')

    data = json.loads(resp.text)

    # check api response status codess
    check_response_code(data)

    detail = data.get("result", {})

    place_identifier = "{} {}".format(detail.get("name"), detail.get("formatted_address"))

    detail_json = {
        "id": detail.get("place_id"),
        "name": detail.get("name"),
        "address": detail.get("formatted_address"),
        "types": detail.get("types"),
        "coordinates": detail.get("geometry", {}).get("location")
    }

    detail_json = add_optional_parameters(
        detail_json, detail,
        *get_populartimes_from_search(place_identifier, **kwargs)
    )

    return detail_json