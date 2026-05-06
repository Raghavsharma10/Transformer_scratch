def generate_join_docs_list(self, left_collection_list, right_collection_list):
        """
            Helper function for merge_join_docs
            :param left_collection_list: Left Collection to be joined
            :type  left_collection_list: MongoCollection

            :param right_collection_list: Right Collection to be joined
            :type  right_collection_list: MongoCollection

            :return joined_docs: List of docs post join
        """

        joined_docs = []
        if (len(left_collection_list) != 0) and (len(right_collection_list) != 0):
            for left_doc in left_collection_list:
                for right_doc in right_collection_list:
                    l_dict = self.change_dict_keys(left_doc, 'L_')
                    r_dict = self.change_dict_keys(right_doc, 'R_')
                    joined_docs.append(dict(l_dict, **r_dict))
        elif left_collection_list:
            for left_doc in left_collection_list:
                joined_docs.append(self.change_dict_keys(left_doc, 'L_'))
        else:
            for right_doc in right_collection_list:
                joined_docs.append(self.change_dict_keys(right_doc, 'R_'))

        return joined_docs