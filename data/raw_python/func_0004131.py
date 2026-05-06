def create_from_name_and_dictionary(self, name, datas):
        """Return a populated object Category from dictionary datas
        """
        category = ObjectCategory(name)
        self.set_common_datas(category, name, datas)

        if "order" in datas:
            category.order = int(datas["order"])

        return category