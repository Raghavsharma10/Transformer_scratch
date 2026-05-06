def start(self, skip_choose=False, fixed_workspace_dir=None):
        """
            Start the application.

            Looks for workspace_location persistent string. If it doesn't find it, uses a default
            workspace location.

            Then checks to see if that workspace exists. If not and if skip_choose has not been
            set to True, asks the user for a workspace location. User may choose new folder or
            existing location. This works by putting up the dialog which will either call start
            again or exit.

            Creates workspace in location if it doesn't exist.

            Migrates database to latest version.

            Creates document model, resources path, etc.
        """
        logging.getLogger("migration").setLevel(logging.INFO)
        if fixed_workspace_dir:
            workspace_dir = fixed_workspace_dir
        else:
            documents_dir = self.ui.get_document_location()
            workspace_dir = os.path.join(documents_dir, "Nion Swift Libraries")
            workspace_dir = self.ui.get_persistent_string("workspace_location", workspace_dir)
        welcome_message_enabled = fixed_workspace_dir is None
        profile, is_created = Profile.create_profile(pathlib.Path(workspace_dir), welcome_message_enabled, skip_choose)
        if not profile:
            self.choose_library()
            return True
        self.workspace_dir = workspace_dir
        DocumentModel.DocumentModel.computation_min_period = 0.1
        document_model = DocumentModel.DocumentModel(profile=profile)
        document_model.create_default_data_groups()
        document_model.start_dispatcher()
        # parse the hardware aliases file
        alias_path = os.path.join(self.workspace_dir, "aliases.ini")
        HardwareSource.parse_hardware_aliases_config_file(alias_path)
        # create the document controller
        document_controller = self.create_document_controller(document_model, "library")
        if self.__resources_path is not None:
            document_model.create_sample_images(self.__resources_path)
        workspace_history = self.ui.get_persistent_object("workspace_history", list())
        if workspace_dir in workspace_history:
            workspace_history.remove(workspace_dir)
        workspace_history.insert(0, workspace_dir)
        self.ui.set_persistent_object("workspace_history", workspace_history)
        self.ui.set_persistent_string("workspace_location", workspace_dir)
        if welcome_message_enabled:
            logging.info("Welcome to Nion Swift.")
        if is_created and len(document_model.display_items) > 0:
            document_controller.selected_display_panel.set_display_panel_display_item(document_model.display_items[0])
            document_controller.selected_display_panel.perform_action("set_fill_mode")
        return True