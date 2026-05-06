def __accept_reject(self, prompt, accepted_text, rejected_text, display_rejected):
        """Return a boolean value for accept/reject."""
        accept_event = threading.Event()
        result_ref = [False]

        def perform():
            def accepted():
                result_ref[0] = True
                accept_event.set()

            def rejected():
                result_ref[0] = False
                accept_event.set()

            self.__message_column.remove_all()
            pose_confirmation_message_box(self.ui, self.__message_column, prompt, accepted, rejected, accepted_text, rejected_text, display_rejected)
            #self.__message_column.add(self.__make_cancel_row())

        with self.__lock:
            self.__q.append(perform)
            self.document_controller.add_task("ui_" + str(id(self)), self.__handle_output_and_q)
        accept_event.wait()
        def update_message_column():
            self.__message_column.remove_all()
            self.__message_column.add(self.__make_cancel_row())
        self.document_controller.add_task("ui_" + str(id(self)), update_message_column)
        return result_ref[0]