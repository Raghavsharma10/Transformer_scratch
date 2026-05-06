def transaction_context(self):
        """Return a context object for a document-wide transaction."""
        class DocumentModelTransaction:
            def __init__(self, document_model):
                self.__document_model = document_model

            def __enter__(self):
                self.__document_model.persistent_object_context.enter_write_delay(self.__document_model)
                return self

            def __exit__(self, type, value, traceback):
                self.__document_model.persistent_object_context.exit_write_delay(self.__document_model)
                self.__document_model.persistent_object_context.rewrite_item(self.__document_model)

        return DocumentModelTransaction(self)