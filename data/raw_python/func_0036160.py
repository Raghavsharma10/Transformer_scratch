def merge_kitchens_improved(dk_api, from_kitchen, to_kitchen):
        """
        returns a string.
        :param dk_api: -- api object
        :param from_kitchen: string
        :param to_kitchen: string  -- kitchen name, string
        :rtype: DKReturnCode
        """
        unresolved_conflicts = DKKitchenDisk.get_unresolved_conflicts(from_kitchen, to_kitchen)
        if unresolved_conflicts is not None and len(unresolved_conflicts) != 0:
            msg = DKCloudCommandRunner._print_unresolved_conflicts(unresolved_conflicts)
            rc = DKReturnCode()
            rc.set(DKReturnCode.DK_FAIL, msg)
            return rc

        resolved_conflicts = DKKitchenDisk.get_resolved_conflicts(from_kitchen, to_kitchen)
        # if resolved_conflicts is not None and len(resolved_conflicts) != 0:

        md = dk_api.merge_kitchens_improved(from_kitchen, to_kitchen, resolved_conflicts)
        if not md.ok():
            md.set_message('merge_kitchens_improved error from %s to Kitchen %s\nmessage: %s' %
                           (from_kitchen, to_kitchen, md.get_message()))
            return md
        merge_no_conflicts = DKCloudCommandRunner._check_no_merge_conflicts(md.get_payload())
        if merge_no_conflicts:
            msg = DKCloudCommandRunner._print_merge_success(md.get_payload())
            current_kitchen = DKKitchenDisk.find_kitchen_name()
            md.set_message(msg)
        else:
            # Found conflicts
            recipe_name = DKRecipeDisk.find_recipe_name()
            kitchen_name = DKKitchenDisk.find_kitchen_name()
            if recipe_name is None and kitchen_name is None:
                # We are not in a kitchen or recipe folder, so just report the findings
                rs = DKCloudCommandRunner.print_merge_conflicts(md.get_payload())
                md.set_message(rs)
            else:
                # We are in a recipe folder, so let's write out the conflicted files.
                rc = DKCloudCommandRunner.write_merge_conflicts(md.get_payload())
                if rc.ok():
                    md.set_message(rc.get_message())
                else:
                    md = rc
        return md