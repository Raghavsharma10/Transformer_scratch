def _ask_to_confirm(self, ui, pac_man, *to_install):
        """ Return True if user wants to install packages, False otherwise """
        ret = DialogHelper.ask_for_package_list_confirm(
            ui, prompt=pac_man.get_perm_prompt(to_install),
            package_list=to_install,
        )
        return bool(ret)