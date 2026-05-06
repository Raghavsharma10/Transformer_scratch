def parse(text):
    """
    Parses the dependency schema from a given string
    (typically a container stdout log)
    """
    found = re.findall(r'(?<=BEGIN_DEPENDENCIES_SCHEMA_OUTPUT>).*(?=<END_DEPENDENCIES_SCHEMA_OUTPUT)', text)

    dependency_results = []

    for match in found:
        data = json.loads(match)

        validate(data)  # will throw ValidationError if invalid

        dependency_results += data['dependencies']

    # we don't have any other fields yet, but in the future
    # may have schema 'version' in which case we'd want to check
    # the versions and compile all the results into 1 schema?

    combined_results = {'dependencies': dependency_results}

    return combined_results