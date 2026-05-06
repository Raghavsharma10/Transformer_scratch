def set_entity_info(self, chain_indices, sequence, description, entity_type):
        """Set the entity level information for the structure.
        :param chain_indices: the indices of the chains for this entity
        :param sequence: the one letter code sequence for this entity
        :param description: the description for this entity
        :param entity_type: the entity type (polymer,non-polymer,water)
        """
        self.entity_list.append(make_entity_dict(chain_indices,sequence,description,entity_type))