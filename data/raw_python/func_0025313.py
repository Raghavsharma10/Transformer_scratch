def library(self) -> Library:
        """Return the library object.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        assert self.__app.document_model
        return Library(self.__app.document_model)