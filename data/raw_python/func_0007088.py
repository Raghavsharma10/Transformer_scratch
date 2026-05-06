def set_data(self, *args):
        """we cant to call set_data to manually update"""
        db = self.begining.get_data() or formats.DATE_DEFAULT
        df = self.end.get_data() or formats.DATE_DEFAULT
        jours = max((df - db).days + 1, 0)
        self.setText(str(jours) + (jours >= 2 and " jours" or " jour"))