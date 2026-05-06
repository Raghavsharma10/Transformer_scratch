def _init_map(self, record_types=None, **kwargs):
        """Initialize form map"""

        osid_objects.OsidSourceableForm._init_map(self)
        osid_objects.OsidObjectForm._init_map(self, record_types=record_types)
        self._my_map['copyrightRegistration'] = self._copyright_registration_default
        self._my_map['assignedRepositoryIds'] = [str(kwargs['repository_id'])]
        self._my_map['copyright'] = self._copyright_default
        self._my_map['title'] = self._title_default
        self._my_map['distributeVerbatim'] = self._distribute_verbatim_default
        self._my_map['createdDate'] = self._created_date_default
        self._my_map['distributeAlterations'] = self._distribute_alterations_default
        self._my_map['principalCreditString'] = self._principal_credit_string_default
        self._my_map['publishedDate'] = self._published_date_default
        self._my_map['sourceId'] = self._source_default
        self._my_map['providerLinkIds'] = self._provider_links_default
        self._my_map['publicDomain'] = self._public_domain_default
        self._my_map['distributeCompositions'] = self._distribute_compositions_default
        self._my_map['compositionId'] = self._composition_default
        self._my_map['published'] = self._published_default
        self._my_map['assetContents'] = []