def find_all_item_classes(self):
        '''Finds and stores a reference to all Redmine_Item subclasses for later use.'''
        # This is a circular import, but performed after the class is defined and an object is instatiated.
        # We do this in order to get references to any objects definitions in the redmine.py file
        # without requiring anyone editing the file to do anything other than create a class with the proper name.
        import redmine as public_classes

        item_class = {}
        for key, value in public_classes.__dict__.items():
            try:
                if issubclass(value, Redmine_Item):
                    item_class[key.lower()] = value
            except:
                continue
        self.item_class = item_class