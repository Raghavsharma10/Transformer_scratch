def get_as_csv(self, output_file_path: Optional[str] = None) -> str:
        """
        Returns the table object as a CSV string.

        :param output_file_path: The output file to save the CSV to, or None.
        :return: CSV representation of the table.
        """
        output = StringIO() if not output_file_path else open(output_file_path, 'w')
        try:
            csv_writer = csv.writer(output)

            csv_writer.writerow(self.columns)
            for row in self.rows:
                csv_writer.writerow(row)
            output.seek(0)
            return output.read()
        finally:
            output.close()