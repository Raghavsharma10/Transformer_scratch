def write_to(self, f):
        """
        Generates code based on the given module configuration and writes it to
        the file object `f`.
        """
        f = CodeWriter(f)

        # Write all header files
        headers = set()
        for plugin in self.plugins:
            headers = headers.union(plugin.header_files())
        for header in headers:
            f.writeln("#include <{}>".format(header))
        f.writeln("")

        # Write all declarations
        for plugin in self.plugins:
            plugin.write_declarations(f)
        f.writeln("")

        # Write the setup function
        with f._function("void", "setup"):
            # Setup all plugins
            f.writeln("// Setup all plugins")
            for plugin in self.plugins:
                plugin.setup_plugin(f)
            # Setup all modules
            f.writeln("// Setup all modules")
            for mod_name in self.modules.keys():
                for plugin in self.plugins:
                    plugin.setup_module(mod_name, f)
        f.writeln("")

        # Write the loop function
        with f._function("void", "loop"):
            # Update all plugins
            f.writeln("// Update all plugins")
            for plugin in self.plugins:
                plugin.update_plugin(f)
            # Update all modules
            f.writeln("// Update all modules")
            for mod_name, mod_info in self.modules.items():
                for plugin in self.plugins:
                    plugin.update_module(mod_name, f)

                # Read all module outputs
                for output_name in mod_info["outputs"]:
                    cond = "{mod_name}.get_{output_name}({msg_name})".format(
                        mod_name=mod_name, output_name=output_name,
                        msg_name=self.msg_name(mod_name, output_name)
                    )
                    with f._if(cond):
                        for plugin in self.plugins:
                            plugin.on_output(mod_name, output_name, f)

            # Read statuses of all modules
            f.writeln("// Read statuses of all modules")
            with f._if("should_read_statuses()"):
                for plugin in self.plugins:
                    plugin.start_read_module_status(f)
                for mod_name in self.modules:
                    for plugin in self.plugins:
                        plugin.read_module_status(mod_name, f)
                for plugin in self.plugins:
                    plugin.end_read_module_status(f)