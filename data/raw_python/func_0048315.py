def get_first_part_id_for_assessment(assessment_id, runtime=None, proxy=None, create=False, bank_id=None):
    """Gets the first part id, which represents the first section, of assessment"""
    if create and bank_id is None:
        raise NullArgument('Bank Id must be provided for create option')
    try:
        return get_next_part_id(assessment_id, runtime, proxy, sequestered=False)[0]
    except IllegalState:
        if create:
            return create_first_assessment_section(assessment_id, runtime, proxy, bank_id)
        else:
            raise