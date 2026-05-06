def save(self, *objs, condition=None, atomic=False):
        """Save one or more objects.

        :param objs: objects to save.
        :param condition: only perform each save if this condition holds.
        :param bool atomic: only perform each save if the local and DynamoDB versions of the object match.
        :raises bloop.exceptions.ConstraintViolation: if the condition (or atomic) is not met.
        """
        objs = set(objs)
        validate_not_abstract(*objs)
        for obj in objs:
            self.session.save_item({
                "TableName": self._compute_table_name(obj.__class__),
                "Key": dump_key(self, obj),
                **render(self, obj=obj, atomic=atomic, condition=condition, update=True)
            })
            object_saved.send(self, engine=self, obj=obj)
        logger.info("successfully saved {} objects".format(len(objs)))