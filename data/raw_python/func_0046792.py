def get_item_bank_id_metadata(self):
        """get the metadata for item bank"""
        metadata = dict(self._item_bank_id_metadata)
        metadata.update({'existing_id_values': self.my_osid_object_form._my_map['itemBankId']})
        return Metadata(**metadata)