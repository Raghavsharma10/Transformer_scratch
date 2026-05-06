def upload_applications(self, metadata, category=None):
        """
        Mimics get starter-kit and wizard functionality to create components
        Note: may create component duplicates, not idempotent
        :type metadata: str
        :type category: Category
        :param metadata: url to meta.yml
        :param category: category
        """
        upload_json = self._router.get_upload(params=dict(metadataUrl=metadata)).json()
        manifests = [dict(name=app['name'], manifest=app['url']) for app in upload_json['applications']]
        if not category:
            category = self.categories['Application']
        data = {'categoryId': category.id, 'applications': manifests}
        self._router.post_application_kits(org_id=self.organizationId, data=json.dumps(data))