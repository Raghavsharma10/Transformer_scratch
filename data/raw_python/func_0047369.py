def add_item(self, *args, **kwargs):
        """Pass through to provider methods."""
        try:
            self._get_provider_session('assessment_basic_authoring_session').add_item(*args, **kwargs)
        except InvalidArgument:
            self._get_sub_package_provider_session(
                'assessment_authoring', 'assessment_part_item_design_session').add_item(*args, **kwargs)