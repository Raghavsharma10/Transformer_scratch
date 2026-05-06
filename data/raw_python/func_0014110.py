def post_post():
    """Create a new post.
    Form Data: title, content, authorid.
    """
    authorid = request.form.get('authorid', None)
    Post(request.form['title'],
         request.form['content'],
         users[authorid])
    return redirect("/posts")