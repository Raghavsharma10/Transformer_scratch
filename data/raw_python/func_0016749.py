def delete(self, *objs, condition=None, atomic=False):
        """Delete one or more objects.

        :param objs: objects to delete.
        :param condition: only perform each delete if this condition holds.
        :param bool atomic: only perform each delete if the local and DynamoDB versions of the object match.
        :raises bloop.exceptions.ConstraintViolation: if the condition (or atomic) is not met.
        """
        objs = set(objs)
        validate_not_abstract(*objs)
        for obj in objs:
            self.session.delete_item({
                "TableName": self._compute_table_name(obj.__class__),
                "Key": dump_key(self, obj),
                **render(self, obj=obj, atomic=atomic, condition=condition)
            })
            object_deleted.send(self, engine=self, obj=obj)
        logger.info("successfully deleted {} objects".format(len(objs)))