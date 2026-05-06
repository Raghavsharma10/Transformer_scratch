def new_workspace(self, name=None, layout=None, workspace_id=None, index=None) -> WorkspaceLayout.WorkspaceLayout:
        """ Create a new workspace, insert into document_model, and return it. """
        workspace = WorkspaceLayout.WorkspaceLayout()
        self.document_model.insert_workspace(index if index is not None else len(self.document_model.workspaces), workspace)
        d = create_image_desc()
        d["selected"] = True
        workspace.layout = layout if layout is not None else d
        workspace.name = name if name is not None else _("Workspace")
        if workspace_id:
            workspace.workspace_id = workspace_id
        return workspace