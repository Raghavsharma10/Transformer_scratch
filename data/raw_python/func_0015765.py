def decorate_with_metadata(obj, result):
    """Return obj decorated with es_meta object"""
    # Create es_meta object with Elasticsearch metadata about this
    # search result
    obj.es_meta = Metadata(
        # Elasticsearch id
        id=result.get('_id', 0),
        # Source data
        source=result.get('_source', {}),
        # The search result score
        score=result.get('_score', None),
        # The document type
        type=result.get('_type', None),
        # Explanation of score
        explanation=result.get('_explanation', {}),
        # Highlight bits
        highlight=result.get('highlight', {})
    )
    # Put the id on the object for convenience
    obj._id = result.get('_id', 0)
    return obj