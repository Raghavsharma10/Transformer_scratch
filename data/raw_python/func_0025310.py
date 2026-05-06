def create_data_and_metadata_io_handler(self, io_handler_delegate):
        """Create an I/O handler that reads and writes a single data_and_metadata.

        :param io_handler_delegate: A delegate object :py:class:`DataAndMetadataIOHandlerInterface`

        .. versionadded:: 1.0

        Scriptable: No
        """
        class DelegateIOHandler(ImportExportManager.ImportExportHandler):
            def __init__(self):
                super().__init__(io_handler_delegate.io_handler_id, io_handler_delegate.io_handler_name, io_handler_delegate.io_handler_extensions)

            def read_data_elements(self, ui, extension, file_path):
                data_and_metadata = io_handler_delegate.read_data_and_metadata(extension, file_path)
                data_element = ImportExportManager.create_data_element_from_extended_data(data_and_metadata)
                return [data_element]

            def can_write(self, data_and_metadata, extension):
                return io_handler_delegate.can_write_data_and_metadata(data_and_metadata, extension)

            def write_display_item(self, ui, display_item: DisplayItemModule.DisplayItem, file_path: str, extension: str) -> None:
                data_item = display_item.data_item
                if data_item:
                    self.write_data_item(ui, data_item, file_path, extension)

            def write_data_item(self, ui, data_item, file_path, extension):
                data_and_metadata = data_item.xdata
                data = data_and_metadata.data
                if data is not None:
                    if hasattr(io_handler_delegate, "write_data_item"):
                        io_handler_delegate.write_data_item(DataItem(data_item), file_path, extension)
                    else:
                        assert hasattr(io_handler_delegate, "write_data_and_metadata")
                        io_handler_delegate.write_data_and_metadata(data_and_metadata, file_path, extension)

        class IOHandlerReference:

            def __init__(self):
                self.__io_handler_delegate = io_handler_delegate
                self.__io_handler = DelegateIOHandler()
                ImportExportManager.ImportExportManager().register_io_handler(self.__io_handler)

            def __del__(self):
                self.close()

            def close(self):
                if self.__io_handler_delegate:
                    io_handler_delegate_close_fn = getattr(self.__io_handler_delegate, "close", None)
                    if io_handler_delegate_close_fn:
                       io_handler_delegate_close_fn()
                    ImportExportManager.ImportExportManager().unregister_io_handler(self.__io_handler)
                    self.__io_handler_delegate = None

        return IOHandlerReference()