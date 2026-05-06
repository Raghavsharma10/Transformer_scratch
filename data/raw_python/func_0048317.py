def get_assessment_section(section_id, runtime=None, proxy=None):
    """Gets a Section given a section_id"""
    from .mixins import LoadedSection
    collection = JSONClientValidated('assessment',
                                     collection='AssessmentSection',
                                     runtime=runtime)
    result = collection.find_one(dict({'_id': ObjectId(section_id.get_identifier())}))
    return LoadedSection(osid_object_map=result, runtime=runtime, proxy=proxy)