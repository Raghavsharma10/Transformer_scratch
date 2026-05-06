def _init_groups(self):
        """
        初始化group数据
        :return:
        """
        for group_id, conf in self.group_conf.items():
            self.parent_input_dict[group_id] = Queue(conf.get('input_max_size', 0))
            self.parent_output_dict[group_id] = Queue(conf.get('output_max_size', 0))