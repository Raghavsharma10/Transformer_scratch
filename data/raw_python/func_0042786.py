def _get_entity(entity_id):
        """
        Uses the request context to retrieve a :class:`meteorpi_model.CameraStatus`, :class:`meteorpi_model.Event` or
        :class:`meteorpi_model.FileRecord` from the POSTed JSON string.

        :param string entity_id:
            The ID of a CameraStatus, Event or FileRecord contained within the request
        :return:
            The corresponding entity from the request.
        """
        entity_type = g.request_dict['type']
        if entity_type == 'file':
            return model.FileRecord.from_dict(g.request_dict['file'])
        elif entity_type == 'metadata':
            return model.ObservatoryMetadata.from_dict(g.request_dict['metadata'])
        elif entity_type == 'observation':
            return model.Observation.from_dict(g.request_dict['observation'])
        else:
            return None