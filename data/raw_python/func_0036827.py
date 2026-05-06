def _replace_placeholder(self, spider):
        """
        Returns replaced db_name and collection_name(base on spider's name).
        if your db_name or collection_name does not have a placeholder or
        your db_name or collection_name that not base on spider's name
        you must override this function.
        """
        return self.db_name % {'spider': spider.name}, self.collection_name % {'spider': spider.name}