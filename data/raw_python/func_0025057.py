def get_string(self, prompt, default_str=None) -> str:
        """Return a string value that the user enters. Raises exception for cancel."""
        accept_event = threading.Event()
        value_ref = [None]

        def perform():
            def accepted(text):
                value_ref[0] = text
                accept_event.set()

            def rejected():
                accept_event.set()

            self.__message_column.remove_all()
            pose_get_string_message_box(self.ui, self.__message_column, prompt, str(default_str), accepted, rejected)
            #self.__message_column.add(self.__make_cancel_row())

        with self.__lock:
            self.__q.append(perform)
            self.document_controller.add_task("ui_" + str(id(self)), self.__handle_output_and_q)
        accept_event.wait()
        def update_message_column():
            self.__message_column.remove_all()
            self.__message_column.add(self.__make_cancel_row())
        self.document_controller.add_task("ui_" + str(id(self)), update_message_column)
        if value_ref[0] is None:
            raise Exception("Cancel")
        return value_ref[0]