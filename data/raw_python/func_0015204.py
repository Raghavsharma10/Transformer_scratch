def generate_menu(self, ass, text, path=None, level=0):
        """
        Function generates menu from based on ass parameter
        """
        menu = self.create_menu()
        for index, sub in enumerate(sorted(ass[1], key=lambda y: y[0].fullname.lower())):
            if index != 0:
                text += "|"
            text += "- " + sub[0].fullname
            new_path = list(path)
            if level == 0:
                new_path.append(ass[0].name)
            new_path.append(sub[0].name)
            menu_item = self.menu_item(sub, new_path)
            if sub[1]:
                # If assistant has subassistants
                (sub_menu, txt) = self.generate_menu(sub, text, new_path, level=level + 1)
                menu_item.set_submenu(sub_menu)
            menu.append(menu_item)
        return menu, text