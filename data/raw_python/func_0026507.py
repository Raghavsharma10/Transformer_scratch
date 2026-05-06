def cmdmap(xdot):
    """Generates a command map"""
    # TODO: Integrate the output into documentation

    from copy import copy

    def print_commands(command, map_output, groups=None, depth=0):
        if groups is None:
            groups = []
        if 'commands' in command.__dict__:
            if len(groups) > 0:
                if xdot:
                    line = "    %s -> %s [weight=1.0];\n" % (groups[-1], command.name)
                else:
                    line = "    " * (depth - 1) + "%s %s\n" % (groups[-1], command.name)
                map_output.append(line)

            for item in command.commands.values():
                subgroups = copy(groups)
                subgroups.append(command.name)
                print_commands(item, map_output, subgroups, depth + 1)
        else:
            if xdot:
                line = "    %s -> %s [weight=%1.1f];\n" % (groups[-1], command.name, len(groups))
            else:
                line = "    " * (len(groups) - 3 + depth) + "%s %s\n" % (groups[-1], command.name)
            map_output.append(line)

    output = []
    print_commands(cli, output)

    output = [line.replace("cli", "isomer") for line in output]

    if xdot:
        with open('iso.dot', 'w') as f:
            f.write('strict digraph {\n')
            f.writelines(sorted(output))
            f.write('}')

        run_process('.', ['xdot', 'iso.dot'])
    else:
        print("".join(output))