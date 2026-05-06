def delete_question(self, question_id):
        """Deletes the ``Question`` identified by the given ``Id``.

        arg:    question_id (osid.id.Id): the ``Id`` of the ``Question``
                to delete
        raise:  NotFound - a ``Question`` was not found identified by
                the given ``Id``
        raise:  NullArgument - ``question_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        collection = JSONClientValidated('assessment',
                                         collection='Item',
                                         runtime=self._runtime)
        if not isinstance(question_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        item = collection.find_one({'question._id': ObjectId(question_id.get_identifier())})

        item['question'] = None
        collection.save(item)