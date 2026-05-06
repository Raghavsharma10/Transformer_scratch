def create_asset(self,
                     asset_data=None,
                     asset_type=None,
                     asset_content_type=None,
                     asset_content_record_types=None,
                     display_name='',
                     description=''):
        """stub"""
        # This method creates a new Asset in the Repository orchestrated
        # with this AssessmentBank:
        return self._set_asset(asset_data=asset_data,
                               asset_type=asset_type,
                               asset_content_type=asset_content_type,
                               asset_content_record_types=asset_content_record_types,
                               display_name=display_name,
                               description=description)