def get(self, table_name):
        """Load table class by name, class not yet initialized"""
        assert table_name in self.tabs, \
            "Table not avaiable. Avaiable tables: {}".format(
                ", ".join(self.tabs.keys())
            )
        return self.tabs[table_name]