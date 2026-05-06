def tableToTsv(self, model):
        """
        Takes a model class and attempts to create a table in TSV format
        that can be imported into a spreadsheet program.
        """
        first = True
        for item in model.select():
            if first:
                header = "".join(
                    ["{}\t".format(x) for x in model._meta.fields.keys()])
                print(header)
                first = False
            row = "".join(
                ["{}\t".format(
                    getattr(item, key)) for key in model._meta.fields.keys()])
            print(row)