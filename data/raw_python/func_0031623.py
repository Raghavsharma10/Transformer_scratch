def load_ipython_extension(ip):
    """
    register magics function, can be called from a notebook
    """
    #ip = get_ipython()
    ip.register_magics(CustomMagics)
    # enable C# (CSHARP) highlight
    patch = ("IPython.config.cell_magic_highlight['clrmagic'] = "
             "{'reg':[/^%%CS/]};")
    js = display.Javascript(data=patch,
                            lib=["https://github.com/codemirror/CodeMirror/blob/master/mode/clike/clike.js"])