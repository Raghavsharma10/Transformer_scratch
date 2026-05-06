def get_html_filename(filename):
    """Converts the filename to a .html extension"""
    if ".html" in filename:
        newFilename = filename
    elif ".md" in filename:
        newFilename = filename.replace(".md", ".html")
    elif ".tile" in filename:
        newFilename = filename.replace(".tile", ".html")
    elif ".jade" in filename:
        newFilename = filename.replace(".jade", ".html")
    elif ".txt" in filename:
        newFilename = filename.replace(".txt", ".html")
    elif ".rst" in filename:
        newFilename = filename.replace(".rst", ".html")
    elif ".docx" in filename:
        newFilename = filename.replace(".docx", ".html")
    else:
        print(filename + " is not a valid file type!")

    return newFilename