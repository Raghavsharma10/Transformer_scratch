def set_item_bank_id(self, bank_id):
        """the assessment bank in which to search for items, such as related to an objective"""
        if self.get_item_bank_id_metadata().is_read_only():
            raise NoAccess()
        if not self.my_osid_object_form._is_valid_id(bank_id):
            raise InvalidArgument()
        self.my_osid_object_form._my_map['itemBankId'] = str(bank_id)