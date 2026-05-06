def get_filter_kwargs(self):
        """
        Translates the cleaned data into a dictionary
        that can used to generate the filter removing
        blank values.
        """
        if self.is_valid():
            filter_kwargs = {}
            for field in self.get_filter_fields():
                empty_values = EMPTY_VALUES
                if hasattr(self.fields[field], 'empty_values'):
                    empty_values = self.fields[field].empty_values

                value = self.cleaned_data.get(field)
                if not value in empty_values:
                    if self.search_fields and field in self.search_fields:
                        filter_kwargs["%s__icontains" % field] = value
                    else:
                        filter_kwargs[field] = value
            return filter_kwargs
        else:
            return {}