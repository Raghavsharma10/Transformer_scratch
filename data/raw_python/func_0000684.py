def process_view(self, request, view_func, view_args, view_kwargs):
        """
        Collect data on Class-Based Views
        """

        # Purge data in view method cache
        # Python 3's keys() method returns an iterator, so force evaluation before iterating.
        view_keys = list(VIEW_METHOD_DATA.keys())
        for key in view_keys:
            del VIEW_METHOD_DATA[key]

        self.view_data = {}

        try:
            cbv = view_func.view_class
        except AttributeError:
            cbv = False

        if cbv:

            self.view_data['cbv'] = True
            klass = view_func.view_class
            self.view_data['bases'] = [base.__name__ for base in inspect.getmro(klass)]
            # Inject with drugz

            for member in inspect.getmembers(view_func.view_class):
                # Check that we are interested in capturing data for this method
                # and ensure that a decorated method is not decorated multiple times.
                if member[0] in VIEW_METHOD_WHITEIST and member[0] not in PATCHED_METHODS[klass]:
                    decorate_method(klass, member[0])
                    PATCHED_METHODS[klass].append(member[0])