def add_preview(self, preview_data, file_name):
        """stub"""
        label = 'preview'
        asset_type = PDF_PREVIEW_ASSET_TYPE
        asset_content_type = PDF_ASSET_CONTENT_GENUS_TYPE
        self.add_file(preview_data,
                      label=label,
                      asset_type=asset_type,
                      asset_content_type=asset_content_type,
                      asset_name=file_name,
                      asset_description='A PDF file with rendered LaTeX.')