def display_plots(filebase, directory=None, width=700, height=500, **kwargs):
    """Display a series of plots controlled by sliders. The function relies on Python string format functionality to index through a series of plots."""
    def show_figure(filebase, directory, **kwargs):
        """Helper function to load in the relevant plot for display."""
        filename = filebase.format(**kwargs)
        if directory is not None:
            filename = directory + '/' + filename
        display(HTML("<img src='{filename}'>".format(filename=filename)))
        
    interact(show_figure, filebase=fixed(filebase), directory=fixed(directory), **kwargs)