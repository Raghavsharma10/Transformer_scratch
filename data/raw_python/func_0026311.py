def _entity_to_region_map(self):
        """
        A dict whose keys are the UUIDs (or just IDs, in some cases) of
        entities, and whose values are the `(rx, ry)` coordinates in which that
        entity can be found. This can be used to easily locate particular
        entities inside the world.
        """
        entity_to_region = {}
        for key in self.get_all_keys():
            layer, rx, ry = struct.unpack('>BHH', key)
            if layer != 4:
                continue
            stream = io.BytesIO(self.get(layer, rx, ry))
            num_entities = sbon.read_varint(stream)
            for _ in range(num_entities):
                uuid = sbon.read_string(stream)
                if uuid in entity_to_region:
                    raise ValueError('Duplicate UUID {}'.format(uuid))
                entity_to_region[uuid] = (rx, ry)
        return entity_to_region