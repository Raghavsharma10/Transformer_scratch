def delete_assessment(self, assessment_id):
        """Deletes an ``Assessment``.

        arg:    assessment_id (osid.id.Id): the ``Id`` of the
                ``Assessment`` to remove
        raise:  NotFound - ``assessment_id`` not found
        raise:  NullArgument - ``assessment_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        """Delete all the children AssessmentParts recursively, too"""
        def remove_children_parts(parent_id):
            part_collection = JSONClientValidated('assessment_authoring',
                                                  collection='AssessmentPart',
                                                  runtime=self._runtime)
            if 'assessment.Assessment' in parent_id:
                query = {"assessmentId": parent_id}
            else:
                query = {"assessmentPartId": parent_id}

            # need to account for magic parts ...
            for part in part_collection.find(query):
                part = assessment_authoring_objects.AssessmentPart(osid_object_map=part,
                                                                   runtime=self._runtime,
                                                                   proxy=self._proxy)
                apls = get_assessment_part_lookup_session(runtime=self._runtime,
                                                          proxy=self._proxy)
                apls.use_unsequestered_assessment_part_view()
                apls.use_federated_bank_view()
                part = apls.get_assessment_part(part.ident)
                try:
                    part.delete()
                except AttributeError:
                    part_collection.delete_one({'_id': ObjectId(part.ident.get_identifier())})
                remove_children_parts(str(part.ident))

        if not isinstance(assessment_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        collection = JSONClientValidated('assessment',
                                         collection='AssessmentOffered',
                                         runtime=self._runtime)
        if collection.find({'assessmentId': str(assessment_id)}).count() != 0:
            raise errors.IllegalState('there are still AssessmentsOffered associated with this Assessment')
        collection = JSONClientValidated('assessment',
                                         collection='Assessment',
                                         runtime=self._runtime)
        collection.delete_one({'_id': ObjectId(assessment_id.get_identifier())})
        remove_children_parts(str(assessment_id))