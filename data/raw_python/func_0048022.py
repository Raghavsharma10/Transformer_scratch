def delete_answer(self, answer_id):
        """Deletes the ``Answer`` identified by the given ``Id``.

        arg:    answer_id (osid.id.Id): the ``Id`` of the ``Answer`` to
                delete
        raise:  NotFound - an ``Answer`` was not found identified by the
                given ``Id``
        raise:  NullArgument - ``answer_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.repository.AssetAdminSession.delete_asset_content_template
        from dlkit.abstract_osid.id.primitives import Id as ABCId
        from .objects import Answer
        collection = JSONClientValidated('assessment',
                                         collection='Item',
                                         runtime=self._runtime)
        if not isinstance(answer_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        item = collection.find_one({'answers._id': ObjectId(answer_id.get_identifier())})

        index = 0
        found = False
        for i in item['answers']:
            if i['_id'] == ObjectId(answer_id.get_identifier()):
                answer_map = item['answers'].pop(index)
            index += 1
            found = True
        if not found:
            raise errors.OperationFailed()
        Answer(
            osid_object_map=answer_map,
            runtime=self._runtime,
            proxy=self._proxy)._delete()
        collection.save(item)