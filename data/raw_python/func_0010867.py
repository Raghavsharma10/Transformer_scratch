def convert_to_dict(self):
        """Convert the group object to an appropriate DICT"""
        out_dict = {}
        out_dict["groupName"] = self.group_name
        out_dict["atomNameList"] = self.atom_name_list
        out_dict["elementList"] = self.element_list
        out_dict["bondOrderList"] = self.bond_order_list
        out_dict["bondAtomList"] = self.bond_atom_list
        out_dict["formalChargeList"] = self.charge_list
        out_dict["singleLetterCode"] = self.single_letter_code
        out_dict["chemCompType"] = self.group_type
        return out_dict