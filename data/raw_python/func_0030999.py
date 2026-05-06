def _find_entity(self, entity):
        ''' Find an Entity instance by name. Checks both name and id fields.'''
        if entity in self.entities:
            return self.entities[entity]
        _ent = [e for e in self.entities.values() if e.name == entity]
        if len(_ent) > 1:
            raise ValueError("Entity name '%s' matches %d entities. To "
                             "avoid ambiguity, please prefix the entity "
                             "name with its domain (e.g., 'bids.%s'." %
                             (entity, len(_ent), entity))
        if _ent:
            return _ent[0]

        raise ValueError("No entity '%s' found." % entity)