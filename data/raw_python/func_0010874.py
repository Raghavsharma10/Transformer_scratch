def set_chain_info(self, chain_id, chain_name, num_groups):
        """Set the chain information.
        :param chain_id: the asym chain id from mmCIF
        :param chain_name: the auth chain id from mmCIF
        :param num_groups: the number of groups this chain has
        """
        self.chain_id_list.append(chain_id)
        self.chain_name_list.append(chain_name)
        self.groups_per_chain.append(num_groups)