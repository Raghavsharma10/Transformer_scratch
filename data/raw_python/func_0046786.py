def set_learning_objective_ids(self, learning_objective_ids):
        """the learning objective to find related items for

        This can only be set if there are no items specifically set

        """
        if self.get_learning_objective_ids_metadata().is_read_only():
            raise NoAccess()
        for learning_objective_id in learning_objective_ids:
            if not self.my_osid_object_form._is_valid_id(learning_objective_id):
                raise InvalidArgument()
        if self.my_osid_object_form._my_map['itemIds'][0]:
            raise IllegalState()
        self.my_osid_object_form._my_map['learningObjectiveIds'] = [str(lo) for lo in learning_objective_ids]