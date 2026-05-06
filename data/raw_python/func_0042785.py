def process_request():
        """
        Retrieve a CameraStatus, Event or FileRecord from the request, based on the supplied type and ID. If the type is
        'cached_request' then the ID must be specified in 'cached_request_id' - if this ID is not for an entity in the
        cache this method will return None and clear the cache (this should only happen under conditions where we've
        failed to correctly handle caching, such as a server restart or under extreme load, but will result in the
        server having to re-request a previous value from the exporting party).

        :return:
            A dict containing 'entity' - the entity for this request or None if there was an issue causing an unexpected
            cache miss, and 'entity-id' which will be the UUID of the entity requested.
            The entity corresponding to this request, or None if we had an issue and there was an unexpected cache miss.
        """
        g.request_dict = safe_load(request.get_data())
        entity_type = g.request_dict['type']
        entity_id = g.request_dict[entity_type]['id']
        ImportRequest.logger.debug("Received request, type={0}, id={1}".format(entity_type, entity_id))
        entity = ImportRequest._get_entity(entity_id)
        ImportRequest.logger.debug("Entity with id={0} was {1}".format(entity_id, entity))
        return ImportRequest(entity=entity, entity_id=entity_id)