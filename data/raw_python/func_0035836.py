def save_to_json(self):
        """The method saves DatasetUpload to json from object"""
        requestvalues = {
            'DatasetId': self.dataset,
            'Name': self.name,
            'Description': self.description,
            'Source': self.source,
            'PubDate': self.publication_date,
            'AccessedOn': self.accessed_on,
            'Url': self.dataset_ref,
            'UploadFormatType': self.upload_format_type,
            'Columns': self.columns,
            'FileProperty': self.file_property.__dict__,
            'FlatDSUpdateOptions': self.flat_ds_update_options,
            'Public': self.public
        }
        return json.dumps(requestvalues)