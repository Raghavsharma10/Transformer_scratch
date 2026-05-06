def _save_keyring(self, keyring_dict):
        """Helper to actually write the keyring to Google"""
        import gdata
        result = self.OK
        file_contents = base64.urlsafe_b64encode(pickle.dumps(keyring_dict))
        try:
            if self.docs_entry:
                extra_headers = {'Content-Type': 'text/plain',
                                 'Content-Length': len(file_contents)}
                self.docs_entry = self.client.Put(
                    file_contents,
                    self.docs_entry.GetEditMediaLink().href,
                    extra_headers=extra_headers
                )
            else:
                from gdata.docs.service import DocumentQuery
                # check for existence of folder, create if required
                folder_query = DocumentQuery(categories=['folder'])
                folder_query['title'] = self.collection
                folder_query['title-exact'] = 'true'
                docs = self.client.QueryDocumentListFeed(folder_query.ToUri())
                if docs.entry:
                    folder_entry = docs.entry[0]
                else:
                    folder_entry = self.client.CreateFolder(self.collection)
                file_handle = io.BytesIO(file_contents)
                media_source = gdata.MediaSource(
                    file_handle=file_handle,
                    content_type='text/plain',
                    content_length=len(file_contents),
                    file_name='temp')
                self.docs_entry = self.client.Upload(
                    media_source,
                    self._get_doc_title(),
                    folder_or_uri=folder_entry
                )
        except gdata.service.RequestError as ex:
            try:
                if ex.message['reason'].lower().find('conflict') != -1:
                    result = self.CONFLICT
                else:
                    # Google docs has a bug when updating a shared document
                    # using PUT from any account other that the owner.
                    # It returns an error 400 "Sorry, there was an error saving
                    # the file. Please try again"
                    # *despite* actually updating the document!
                    # Workaround by re-reading to see if it actually updated
                    msg = 'Sorry, there was an error saving the file'
                    if ex.message['body'].find(msg) != -1:
                        new_docs_entry, new_keyring_dict = self._read()
                        if new_keyring_dict == keyring_dict:
                            result = self.OK
                        else:
                            result = self.FAIL
                    else:
                        result = self.FAIL
            except Exception:
                result = self.FAIL

        return result