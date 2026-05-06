def _heavyQuery(variantSetId, callSetIds):
    """
    Very heavy query: calls for the specified list of callSetIds
    on chromosome 2 (11 pages, 90 seconds to fetch the entire thing
    on a high-end desktop machine)
    """
    request = protocol.SearchVariantsRequest()
    request.reference_name = '2'
    request.variant_set_id = variantSetId
    for callSetId in callSetIds:
        request.call_set_ids.add(callSetId)
    request.page_size = 100
    request.end = 100000
    return request