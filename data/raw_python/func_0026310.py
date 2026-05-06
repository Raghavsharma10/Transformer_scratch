def get_entity_uuid_coords(self, uuid):
        """
        Returns the coordinates of the given entity UUID inside this world, or
        `None` if the UUID is not found.
        """
        if uuid in self._entity_to_region_map:
            coords = self._entity_to_region_map[uuid]
            entities = self.get_entities(*coords)
            for entity in entities:
                if 'uniqueId' in entity.data and entity.data['uniqueId'] == uuid:
                    return tuple(entity.data['tilePosition'])
        return None