def _create_component_results(json_data, result_key):
    """ Returns a list of ComponentResult from the json_data"""
    component_results = []
    for key, value in list(json_data.items()):
        if key not in [result_key, "meta"]:
            component_result = ComponentResult(
                key,
                value["result"],
                value["api_code"],
                value["api_code_description"]
            )

            component_results.append(component_result)

    return component_results