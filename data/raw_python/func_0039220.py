def _anonymize_table(cls, table_data, pii_fields):
        """Anonymize in `table_data` the fields in `pii_fields`.

        Args:
            table_data (pandas.DataFrame): Original dataframe/table.
            pii_fields (list[dict]): Metadata for the fields to transform.

        Result:
            pandas.DataFrame: Anonymized table.
        """
        for pii_field in pii_fields:
            field_name = pii_field['name']
            transformer = cls.get_class(TRANSFORMERS['categorical'])(pii_field)
            table_data[field_name] = transformer.anonymize_column(table_data)

        return table_data