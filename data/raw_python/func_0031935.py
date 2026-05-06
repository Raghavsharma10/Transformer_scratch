def get_html_clear_filename(filename):
    """Clears the file extension from the filename and makes it nice looking"""
    newFilename = filename.replace(".html", "")
    newFilename = newFilename.replace(".md", "")
    newFilename = newFilename.replace(".txt", "")
    newFilename = newFilename.replace(".tile", "")
    newFilename = newFilename.replace(".jade", "")
    newFilename = newFilename.replace(".rst", "")
    newFilename = newFilename.replace(".docx", "")
    newFilename = newFilename.replace("index", "home")
    newFilename = newFilename.replace("-", " ")
    newFilename = newFilename.replace("_", " ")
    newFilename = newFilename.title()

    return newFilename