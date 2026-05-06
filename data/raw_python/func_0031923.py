def convert_text(filename):
    """Convert the post/page content using the converters"""
    text_content = open(filename, "r")
    if ".md" in filename:
        text_cont1 = "\n" + markdown.markdown(text_content.read()) + "\n"
    elif ".docx" in filename:
        with open(os.path.join(cwd, "content", filename), "rb") as docx_file:
            result = mammoth.convert_to_html(docx_file)
            final_docx_html = result.value
        text_cont1 = "\n" + final_docx_html + "\n"
    elif ".tile" in filename:
        text_cont1 = "\n" + textile.textile(text_content.read()) + "\n"
    elif ".jade" in filename:
        text_cont1 = "\n" + pyjade.simple_convert(text_content.read()) + "\n"
    elif ".rst" in filename:
        text_cont1 = "\n" + \
            publish_parts(text_content.read(), writer_name='html')[
                'html_body'] + "\n"
    elif ".html" in filename:
        text_cont1 = text_content.read()
    elif ".txt" in filename:
        text_cont1 = text_content.read()
    else:
        print(filename + " is not a valid file type!")
        text_cont1 = "NULL"

    return text_cont1 + "\n\n"