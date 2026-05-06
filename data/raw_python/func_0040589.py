def _report(self, action, key_mapper=mappers._report_key_mapper):
        '''Return the dictionary of **kwargs with the correct datums attribute
        names and data types for the top level of the report, and return the
        nested levels separately.
        '''
        _top_level = [
            k for k, v in self.report.items() if not isinstance(v, dict)]
        _nested_level = [
            k for k, v in self.report.items() if isinstance(v, dict)]
        top_level_dict = {}
        nested_levels_dict = {}
        for key in _top_level:
            try:
                if key == 'date' or key == 'timestamp':
                    item = mappers._key_type_mapper[key](
                        str(self.report[key]), **{'ignoretz': True})
                else:
                    item = mappers._key_type_mapper[key](str(
                        self.report[key]) if key != 'draft' else self.report[key])
            except KeyError:
                item = self.report[key]
            finally:
                try:
                    top_level_dict[key_mapper[key]] = item
                except KeyError:
                    warnings.warn('''
                        {0} is not currently supported by datums and will be ignored.
                        Would you consider submitting an issue to add support?
                        https://www.github.com/thejunglejane/datums/issues
                        '''.format(key))
        for key in _nested_level:
            nested_levels_dict[key] = self.report[key]
            # Add the parent report ID
            nested_levels_dict[key][
                'reportUniqueIdentifier'] = mappers._key_type_mapper[
                    'uniqueIdentifier'](str(self.report['uniqueIdentifier']))
            if key == 'placemark':
                # Add the parent location report UUID
                nested_levels_dict[key][
                    'locationUniqueIdentifier'] = nested_levels_dict[key].pop(
                        'reportUniqueIdentifier')
            # Create UUID for altitude report if there is not one and the action
            # is get_or_create, else delete the altitude report from the nested
            # levels and warn that it will not be updated
            if 'uniqueIdentifier' not in nested_levels_dict[key]:
                if action.__func__.func_name == 'get_or_create':
                    nested_levels_dict[key]['uniqueIdentifier'] = uuid.uuid4()
                else:
                    del nested_levels_dict[key]
                    warnings.warn('''
                        No uniqueIdentifier found for AltitudeReport in {0}.
                        Existing altitude report will not be updated.
                        '''.format(self.report['uniqueIdentifier']))
        return top_level_dict, nested_levels_dict