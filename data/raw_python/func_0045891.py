def get_item(self):
        """Gets the ``Item``.

        return: (osid.assessment.Item) - the assessment item
        *compliance: mandatory -- This method must be implemented.*

        """
        # So, for now we're assuming that what should be returned here is the question.
        # We could change this class impl to "know" if it came from a ResponseLookupSession call
        # and return the whole Item if so.
        try:
            # an un-answered response will have a magic itemId here
            item_lookup_session = get_item_lookup_session(runtime=self._runtime, proxy=self._proxy)
            item_lookup_session.use_federated_bank_view()
            item = item_lookup_session.get_item(self._item_id)
        except errors.NotFound:
            # otherwise an answered response will have an assessment-session itemId
            if self._section is not None:
                question = self._section.get_question(self._item_id)
                ils = self._section._get_item_lookup_session()
                real_item_id = Id(question._my_map['itemId'])
                item = ils.get_item(real_item_id)
            else:
                raise errors.NotFound()
        return item.get_question()