def merge_configs(self, config, datas):
        """Merge configs files
        """
        if not isinstance(config, dict) or len([x for x in datas if not isinstance(x, dict)]) > 0:
            raise TypeError("Unable to merge: Dictionnary expected")

        for key, value in config.items():
            others = [x[key] for x in datas if key in x]
            if len(others) > 0:
                if isinstance(value, dict):
                    config[key] = self.merge_configs(value, others)
                else:
                    config[key] = others[-1]
        return config