def uniquify_by(self, column, chooser=None, aggregate='MAX'):
        """
        Group by `column` and run `aggregate` function on `chooser` column.
        """
        self.group_by.append(column)
        if chooser:
            i = self.columns.index(chooser)
            self.columns[i] = '{0}({1})'.format(aggregate, self.columns[i])