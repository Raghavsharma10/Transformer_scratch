def load(self):
        """
        Extract tabular data as |TableData| instances from a Line-delimited JSON file.
        |load_source_desc_file|

        :return:
            Loaded table data iterator.
            |load_table_name_desc|

        :rtype: |TableData| iterator
        :raises pytablereader.DataError:
            If the data is invalid Line-delimited JSON.
        :raises pytablereader.error.ValidationError:
            If the data is not acceptable Line-delimited JSON format.
        """

        formatter = JsonLinesTableFormatter(self.load_dict())
        formatter.accept(self)

        return formatter.to_table_data()