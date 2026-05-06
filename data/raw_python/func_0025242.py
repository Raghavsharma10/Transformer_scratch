def close(self):
        """Close the document controller.

        This method must be called to shut down the document controller. There are several
        paths by which it can be called, though.

           * User quits application via menu item. The menu item will call back to Application.exit which will close each
             document controller by calling this method.
           * User quits application using dock menu item. The Qt application will call aboutToClose in the document windows
           * User closes document window via menu item.
           * User closes document window via close box.

        The main concept of closing is that it is always triggered by the document window closing. This can be initiated
        from within Python by calling request_close on the document window. When the window closes, either by explicit request
        or by the user clicking a close box, it will invoke the about_to_close method on the document window. At this point,
        the window would still be open, so the about_to_close message can be used to tell the document controller to save anything
        it needs to save and prepare for closing.
        """
        assert self.__closed == False
        self.__closed = True
        self.finish_periodic()  # required to finish periodic operations during tests
        # dialogs
        for weak_dialog in self.__dialogs:
            dialog = weak_dialog()
            if dialog:
                try:
                    dialog.request_close()
                except Exception as e:
                    pass
        # menus
        self._file_menu = None
        self._edit_menu = None
        self._processing_menu = None
        self._view_menu = None
        self._window_menu = None
        self._help_menu = None
        self._library_menu = None
        self._processing_arithmetic_menu = None
        self._processing_reduce_menu = None
        self._processing_transform_menu = None
        self._processing_filter_menu = None
        self._processing_fourier_menu = None
        self._processing_graphics_menu = None
        self._processing_sequence_menu = None
        self._processing_redimension_menu = None
        self._display_type_menu = None

        if self.__workspace_controller:
            self.__workspace_controller.close()
            self.__workspace_controller = None
        self.__call_soon_event_listener.close()
        self.__call_soon_event_listener = None
        self.__filtered_display_items_model.close()
        self.__filtered_display_items_model = None
        self.filter_controller.close()
        self.filter_controller = None
        self.__display_items_model.close()
        self.__display_items_model = None
        # document_model may be shared between several DocumentControllers, so use reference counting
        # to determine when to close it.
        self.document_model.remove_ref()
        self.document_model = None
        self.did_close_event.fire(self)
        self.did_close_event = None
        super().close()