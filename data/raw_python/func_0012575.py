async def extract_rows(self, file_or_name, **reader_kwargs):
        """
        Extract `self.sample_size` rows from the CSV file and analyze their
        data-types.
        :param str file_or_name: A string filename or a file handle.
        :param reader_kwargs: Arbitrary parameters to pass to the CSV reader.
        :returns: A 2-tuple containing a list of headers and list of rows
                  read from the CSV file.
        """
        rows = []
        rows_to_read = self.sample_size
        async with self.get_reader(file_or_name, **reader_kwargs) as reader:
            if self.has_header:
                rows_to_read += 1
            for i in range(self.sample_size):
                try:
                    row = await reader.__anext__()
                except AttributeError as te:
                    row = next(reader)
                except:
                    raise
                rows.append(row)

        if self.has_header:
            header, rows = rows[0], rows[1:]
        else:
            header = ['field_%d' % i for i in range(len(rows[0]))]
        return header, rows