def convert_ids2objects(self):
        """ Convert object IDs from `self._json_params` to objects if needed.

        Only IDs that belong to relationship field of `self.Model`
        are converted.
        """
        if not self.Model:
            log.info("%s has no model defined" % self.__class__.__name__)
            return

        for field in self._json_params.keys():
            if not engine.is_relationship_field(field, self.Model):
                continue
            rel_model_cls = engine.get_relationship_cls(field, self.Model)
            self.id2obj(field, rel_model_cls)