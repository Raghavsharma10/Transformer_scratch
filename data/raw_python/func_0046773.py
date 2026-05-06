def load_item_for_objective(self):
        """if this is the first time for this magic part, find an LO linked item"""
        mgr = self.my_osid_object._get_provider_manager('ASSESSMENT', local=True)
        if self.my_osid_object._my_map['itemBankId']:
            item_query_session = mgr.get_item_query_session_for_bank(Id(self.my_osid_object._my_map['itemBankId']),
                                                                     proxy=self.my_osid_object._proxy)
        else:
            item_query_session = mgr.get_item_query_session(proxy=self.my_osid_object._proxy)
        item_query_session.use_federated_bank_view()
        item_query = item_query_session.get_item_query()
        for objective_id_str in self.my_osid_object._my_map['learningObjectiveIds']:
            item_query.match_learning_objective_id(Id(objective_id_str), True)
        item_list = list(item_query_session.get_items_by_query(item_query))
        # Let's query all takens and their children sections for questions, to
        # remove seen ones
        taking_agent_id = self._assessment_section._assessment_taken.taking_agent_id
        atqs = mgr.get_assessment_taken_query_session(proxy=self.my_osid_object._proxy)
        atqs.use_federated_bank_view()
        querier = atqs.get_assessment_taken_query()
        querier.match_taking_agent_id(taking_agent_id, match=True)
        # let's seed this with the current section's questions
        seen_items = [item_id for item_id in self._assessment_section._item_id_list]
        taken_ids = [str(t.ident)
                     for t in atqs.get_assessments_taken_by_query(querier)]
        # Try to find the questions directly via Mongo query -- don't do
        # for section in taken._get_assessment_sections():
        #     seen_items += [question['itemId'] for question in section._my_map['questions']]
        # because standing up all the sections is wasteful
        collection = JSONClientValidated('assessment',
                                         collection='AssessmentSection',
                                         runtime=self.my_osid_object._runtime)
        results = collection.find({"assessmentTakenId": {"$in": taken_ids}})
        for section in results:
            if 'questions' in section:
                seen_items += [question['itemId'] for question in section['questions']]
        unseen_item_id = None
        # need to randomly shuffle this item_list
        shuffle(item_list)
        for item in item_list:
            if str(item.ident) not in seen_items:
                unseen_item_id = item.get_id()
                break
        if unseen_item_id is not None:
            self.my_osid_object._my_map['itemIds'] = [str(unseen_item_id)]
        elif self.my_osid_object._my_map['allowRepeatItems']:
            if len(item_list) > 0:
                self.my_osid_object._my_map['itemIds'] = [str(item_list[0].ident)]
            else:
                self.my_osid_object._my_map['itemIds'] = []  # don't put '' here, it will break when it tries to find an item with id ''
        else:
            self.my_osid_object._my_map['itemIds'] = []