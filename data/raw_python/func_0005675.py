def prettify(string):
    """
    replace markup emoji and progressbars with actual things

    # Example
    ```python
    from habitipy.util import prettify
    print(prettify('Write thesis :book: ![progress](http://progressed.io/bar/0 "progress")'))
    ```
    ```
    Write thesis 📖 ██████████0%
    ```
    """
    string = emojize(string, use_aliases=True) if emojize else string
    string = progressed(string)
    return string