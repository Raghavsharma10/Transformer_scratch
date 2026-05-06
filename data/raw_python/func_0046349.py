def get_answers(self):
        """Gets the answers.

        return: (osid.assessment.AnswerList) - the answers
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.repository.Asset.get_asset_contents_template
        return AnswerList(
            self._my_map['answers'],
            runtime=self._runtime,
            proxy=self._proxy)