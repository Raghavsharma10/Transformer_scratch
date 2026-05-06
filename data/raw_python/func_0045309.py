def _handle_uniqueness(self):
        """
        Checks marked as unique and unique_together fields of the Model at each
        creation and update, and if it violates the uniqueness raises IntegrityError.

        First, looks at the fields which marked as "unique". If Model's unique fields
        did not change, it means that there is still a record at db with same unique
        field values. So, it must be checked that if more than one result violates the
        uniqueness. If it is, raise an IntegrityError. Otherwise, when marked as unique
        fields in the list of changed fields, it must be checked that if exists any
        violation instead of more than one. And, if it is, again raise an IntegrityError.

        Then, looks at the fields which marked as "unique_together" with the same logic.

        Raises:
            IntegrityError if unique and unique_together checks does not pass
        """

        def _getattr(u):
            try:
                return self._field_values[u]
            except KeyError:
                return getattr(self, u)

        if self._uniques:
            for u in self._uniques:
                val = _getattr(u)
                changed_fields = self.changed_fields(from_db=True)
                if self.exist and not (u in changed_fields if not callable(val) else
                                       (str(u) + "_id") in changed_fields):
                    if val and self.objects.filter(**{u: val}).count() > 1:
                        raise IntegrityError("Unique mismatch: %s for %s already exists for value: "
                                             "%s" % (u, self.__class__.__name__, val))
                else:
                    if val and self.objects.filter(**{u: val}).count():
                        raise IntegrityError("Unique mismatch: %s for %s already exists for value: "
                                             "%s" % (u, self.__class__.__name__, val))
        if self.Meta.unique_together:
            changed_fields = self.changed_fields(from_db=True)
            for uniques in self.Meta.unique_together:
                vals = dict([(u, _getattr(u)) for u in uniques])
                if self.exist:
                    query_is_changed = []
                    for uni in vals.keys():
                        if callable(vals[uni]):
                            is_changed = (str(uni) + "_id") in changed_fields
                            query_is_changed.append(is_changed)
                        else:
                            is_changed = uni in changed_fields
                            query_is_changed.append(is_changed)
                    is_unique_changed = any(query_is_changed)
                    if not is_unique_changed:
                        if self.objects.filter(**vals).count() > 1:
                            raise IntegrityError(
                                "Unique together mismatch: %s combination already exists for %s"
                                % (vals, self.__class__.__name__))
                    else:
                        if self.objects.filter(**vals).count():
                            raise IntegrityError(
                                "Unique together mismatch: %s combination already exists for %s"
                                % (vals, self.__class__.__name__))
                else:
                    if self.objects.filter(**vals).count():
                        raise IntegrityError(
                            "Unique together mismatch: %s combination already exists for %s"
                            % (vals, self.__class__.__name__))