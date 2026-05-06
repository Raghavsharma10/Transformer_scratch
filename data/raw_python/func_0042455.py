def _handle_next_export_subtask(self, export_state=None):
        """
        Process the next export sub-task, if there is one.

        :param ExportState export_state:
            If provided, this is used instead of the database queue, in effect directing the exporter to process the
            previous export again. This is used to avoid having to query the database when we know already what needs
            to be done. It also maintains a cache of the entity so we don't have to re-acquire it on multiple exports.
        :return:
            A :class:`meteorpi_db.exporter.MeteorExporter.ExportStateCache` representing the state of the export, or
            None if there was nothing to do.
        """
        # Use a cached state, or generate a new one if required
        if export_state is None or export_state.export_task is None:
            export = self.db.get_next_entity_to_export()
            if export is not None:
                export_state = self.ExportState(export_task=export)
            else:
                return None
        try:
            auth = (export_state.export_task.target_user,
                    export_state.export_task.target_password)
            target_url = export_state.export_task.target_url
            response = post(url=target_url, verify=False,
                            json=export_state.entity_dict,
                            auth=auth)
            response.raise_for_status()
            json = response.json()
            state = json['state']
            if state == 'complete':
                return export_state.fully_processed()
            elif state == 'need_file_data':
                file_id = json['file_id']
                file_record = self.db.get_file(repository_fname=file_id)
                if file_record is None:
                    return export_state.failed()
                with open(self.db.file_path_for_id(file_id), 'rb') as file_content:
                    multi = MultipartEncoder(fields={'file': ('file', file_content, file_record.mime_type)})
                    post(url="{0}/data/{1}/{2}".format(target_url, file_id, file_record.file_md5),
                         data=multi, verify=False,
                         headers={'Content-Type': multi.content_type},
                         auth=auth)
                return export_state.partially_processed()
            elif state == 'continue':
                return export_state.partially_processed()
            else:
                return export_state.confused()
        except HTTPError:
            traceback.print_exc()
            return export_state.failed()
        except ConnectionError:
            traceback.print_exc()
            return export_state.failed()