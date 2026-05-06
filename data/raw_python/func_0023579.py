def folder_get_content(self, folder_key=None, content_type=None,
                           filter_=None, device_id=None, order_by=None,
                           order_direction=None, chunk=None, details=None,
                           chunk_size=None):
        """folder/get_content

        http://www.mediafire.com/developers/core_api/1.3/folder/#get_content
        """
        return self.request('folder/get_content', QueryParams({
            'folder_key': folder_key,
            'content_type': content_type,
            'filter': filter_,
            'device_id': device_id,
            'order_by': order_by,
            'order_direction': order_direction,
            'chunk': chunk,
            'details': details,
            'chunk_size': chunk_size
        }))