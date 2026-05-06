def document_windows(self) -> typing.List[DocumentWindow]:
        """Return the document windows.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        return [DocumentWindow(document_controller) for document_controller in self.__application.document_controllers]