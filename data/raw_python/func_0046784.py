def set_item_ids(self, item_ids):
        '''the target Item

        This can only be set if there is no learning objective set

        '''
        if self.get_item_ids_metadata().is_read_only():
            raise NoAccess()
        for item_id in item_ids:
            if not self.my_osid_object_form._is_valid_id(item_id):
                raise InvalidArgument()
        if self.my_osid_object_form._my_map['learningObjectiveIds'][0]:
            raise IllegalState()
        self.my_osid_object_form._my_map['itemIds'] = [str(i) for i in item_ids]