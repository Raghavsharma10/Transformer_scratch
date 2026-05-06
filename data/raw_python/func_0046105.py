def _update_object_map(self, obj_map):
        """loop through all the keys in self.my_osid_object._my_map, and
        see if any of them contain text like "AssetContent:<label>"
        If so, assume it is markup (?), replace the string with asset_content.get_url()"""
        # TODO: Look for <img> tags to add in alt-text and description
        # TODO: Look for <video> and <audio> tags to add in description, transcripts and vtt files?
        try:
            super(FilesRecord, self)._update_object_map(obj_map)
        except AttributeError:
            pass

        bypass_asset_content_authorization = False
        acls = None

        try:
            config = self.my_osid_object._runtime.get_configuration()
            parameter_id = Id('parameter:bypassAuthorizationForFilesRecordAssetContentLookup@json')
            bypass_asset_content_authorization = config.get_value_by_parameter(parameter_id).get_boolean_value()
        except (AttributeError, KeyError, NotFound):
            pass

        def replace_url_in_display_text(potential_display_text, dict_files_map):
            if ('text' in potential_display_text and
                    potential_display_text['text'] is not None and
                    'AssetContent' in potential_display_text['text']):
                # assume markup? Wrap this in case it's not a valid XML doc
                # with a single parent object
                wrapped_text = '<wrapper>{0}</wrapper'.format(potential_display_text['text'])
                soup = BeautifulSoup(wrapped_text, 'xml')
                media_file_elements = soup.find_all(src=media_regex)
                media_file_elements += soup.find_all(data=media_regex)
                for media_file_element in media_file_elements:
                    if 'src' in media_file_element.attrs:
                        media_key = 'src'
                    else:
                        media_key = 'data'
                    if ':' not in media_file_element[media_key]:
                        continue

                    media_label = media_file_element[media_key].split(':')[-1]

                    if media_label in dict_files_map:
                        asset_id = Id(dict_files_map[media_label]['assetId'])
                        ac_id = Id(dict_files_map[media_label]['assetContentId'])
                        if bypass_asset_content_authorization:
                            ac = acls.get_asset_content(ac_id)
                        else:
                            ac = self._get_asset_content(asset_id=asset_id, asset_content_id=ac_id)

                        if media_file_element.name == 'track':
                            try:
                                if not ac.has_files():
                                    continue
                            except AttributeError:
                                # non-multi-language VTT files
                                media_file_element[media_key] = ac.get_url()
                            else:
                                media_file_element[media_key] = ac.get_url()
                                media_file_element['srclang'] = ac.get_vtt_locale_identifier().lower()[0:2]
                                media_file_element['label'] = ac.get_vtt_locale_label()
                        elif media_file_element.name == 'transcript':
                            if not ac.has_files():
                                continue
                            transcript_template_path = '{0}/osid/transcript_template.xml'.format(ABS_PATH)
                            with codecs.open(transcript_template_path, 'r', encoding='utf-8') as template_file:
                                template = template_file.read().format(media_label,
                                                                       ac.get_transcript_locale_label().lower(),
                                                                       ac.get_transcript_locale_label().title(),
                                                                       ac.get_transcript_text())
                                new_template_tag = BeautifulSoup(template, 'xml').div
                                # media_file_element.replace_with(new_template_tag)
                                p_parent = None
                                for parent in media_file_element.parents:
                                    if parent is not None and parent.name != 'p':
                                        # insert the transcript after the top p tag
                                        # so that we don't create invalid HTML by nesting
                                        # <div> and <aside> inside of a <p> tag
                                        p_parent.insert_after(new_template_tag)
                                        break
                                    p_parent = parent
                                media_file_element.extract()
                        else:
                            media_file_element[media_key] = ac.get_url()

                    # check for alt-tags
                    if 'alt' in media_file_element.attrs:
                        alt_tag_label = media_file_element['alt'].split(':')[-1]
                        if alt_tag_label in dict_files_map:
                            asset_id = Id(dict_files_map[alt_tag_label]['assetId'])
                            ac_id = Id(dict_files_map[alt_tag_label]['assetContentId'])
                            if bypass_asset_content_authorization:
                                ac = acls.get_asset_content(ac_id)
                            else:
                                ac = self._get_asset_content(asset_id=asset_id, asset_content_id=ac_id)
                            try:
                                media_file_element['alt'] = ac.get_alt_text().text
                            except AttributeError:
                                pass

                potential_display_text['text'] = soup.wrapper.renderContents().decode('utf-8')
            else:
                for new_key, value in potential_display_text.items():
                    if isinstance(value, list):
                        new_files_map = dict_files_map
                        if 'fileIds' in potential_display_text:
                            new_files_map = potential_display_text['fileIds']
                        potential_display_text[new_key] = check_list_children(value, new_files_map)
            return potential_display_text

        def check_list_children(potential_text_list, list_files_map):
            updated_list = []
            for child in potential_text_list:
                if isinstance(child, dict):
                    files_map = list_files_map
                    if 'fileIds' in child:
                        files_map = child['fileIds']
                    updated_list.append(replace_url_in_display_text(child, files_map))
                elif isinstance(child, list):
                    updated_list.append(check_list_children(child, list_files_map))
                else:
                    updated_list.append(child)
            return updated_list

        if bypass_asset_content_authorization:
            # One assumption is that the object's catalogId can be used
            # as the repositoryId
            manager = self.my_osid_object._get_provider_manager('REPOSITORY')

            try:
                if self.my_osid_object._proxy is not None:
                    acls = manager.get_asset_content_lookup_session(proxy=self.my_osid_object._proxy)
                else:
                    acls = manager.get_asset_content_lookup_session()
            except AttributeError:
                pass
            else:
                acls.use_federated_repository_view()

        media_regex = re.compile('(AssetContent:)')
        original_files_map = {}
        if 'fileIds' in obj_map:
            original_files_map = obj_map['fileIds']

        for key, data in obj_map.items():
            if isinstance(data, dict):
                obj_map[key] = replace_url_in_display_text(data, original_files_map)
            elif isinstance(data, list):
                obj_map[key] = check_list_children(data, original_files_map)