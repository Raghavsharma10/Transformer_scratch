def export_olx(self, tarball, root_path):
        """if sequestered, only export the assets"""
        def append_asset_to_soup_and_export(asset_):
            if isinstance(asset_, Item):
                try:
                    unique_url = asset_.export_olx(tarball, root_path)
                except AttributeError:
                    pass
                else:
                    unique_name = get_file_name_without_extension(unique_url)
                    asset_type = asset_.genus_type.identifier
                    asset_tag = my_soup.new_tag(asset_type)
                    asset_tag['url_name'] = unique_name
                    getattr(my_soup, my_tag).append(asset_tag)
            else:
                try:
                    unique_urls = asset_.export_olx(tarball, root_path)
                except AttributeError:
                    pass
                else:
                    for index, ac in enumerate(asset_.get_asset_contents()):
                        asset_type = ac.genus_type.identifier

                        unique_url = unique_urls[index]
                        unique_name = get_file_name_without_extension(unique_url)
                        asset_tag = my_soup.new_tag(asset_type)

                        asset_tag['url_name'] = unique_name
                        getattr(my_soup, my_tag).append(asset_tag)

        def get_file_name_without_extension(filepath):
            return filepath.split('/')[-1].replace('.xml', '')

        my_path = None
        if self.my_osid_object.is_sequestered():
            # just export assets
            for asset in self.assets:
                try:
                    asset.export_olx(tarball, root_path)
                except AttributeError:
                    pass
        else:
            # also add to the /<tag>/ folder
            my_tag = self.my_osid_object.genus_type.identifier
            expected_name = self.get_unique_name(tarball, self.url, my_tag, root_path)
            my_path = '{0}{1}/{2}.xml'.format(root_path,
                                              my_tag,
                                              expected_name)
            my_soup = BeautifulSoup('<' + my_tag + '/>', 'xml')
            getattr(my_soup, my_tag)['display_name'] = self.my_osid_object.display_name.text

            if my_tag == 'split_test':
                getattr(my_soup, my_tag)['group_id_to_child'] = self.my_osid_object.group_id_to_child
                getattr(my_soup, my_tag)['user_partition_id'] = self.my_osid_object.user_partition_id.text

            rm = self.my_osid_object._get_provider_manager('REPOSITORY')
            if self.my_osid_object._proxy is None:
                cls = rm.get_composition_lookup_session()
            else:
                cls = rm.get_composition_lookup_session(proxy=self.my_osid_object._proxy)
            cls.use_federated_repository_view()
            cls.use_unsequestered_composition_view()
            for child_id in self.my_osid_object.get_child_ids():
                child = cls.get_composition(child_id)
                if child.is_sequestered():
                    # append its assets here
                    for asset in child.assets:
                        append_asset_to_soup_and_export(asset)
                else:
                    child_type = child.genus_type.identifier
                    child_tag = my_soup.new_tag(child_type)

                    child_path = child.export_olx(tarball, root_path)
                    if child_path is not None:
                        child_tag['url_name'] = get_file_name_without_extension(child_path)
                    getattr(my_soup, my_tag).append(child_tag)

            for asset in self.assets:
                append_asset_to_soup_and_export(asset)

            self.write_to_tarfile(tarball, my_path, my_soup)

        return my_path