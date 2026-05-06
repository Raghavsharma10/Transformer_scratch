def get_learning_objective_ids(self):
        """ This method mirrors that in the Item.

        So that questions can also be inspected for learning objectives

        """
        if 'learningObjectiveIds' not in self._my_map:  # Will this ever be the case?
            collection = JSONClientValidated('assessment',
                                             collection='Item',
                                             runtime=self._runtime)
            item_map = collection.find_one({'_id': ObjectId(Id(self._my_map['itemId']).get_identifier())})
            self._my_map['learningObjectiveIds'] = list(item_map['learningObjectiveIds'])
        return IdList(self._my_map['learningObjectiveIds'])