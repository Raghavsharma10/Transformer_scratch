def nestSections(block, level=1):
    """
    Sections aren't handled by CommonMark at the moment.
    This function adds sections to a block of nodes.
    'title' nodes with an assigned level below 'level' will be put in a child section.
    If there are no child nodes with titles of level 'level' then nothing is done
    """
    cur = block.first_child
    if cur is not None:
        children = []
        # Do we need to do anything?
        nest = False
        while cur is not None:
            if cur.t == 'heading' and cur.level == level:
                nest = True
                break
            cur = cur.nxt
        if not nest:
            return

        section = Node('MDsection', 0)
        section.parent = block
        cur = block.first_child
        while cur is not None:
            if cur.t == 'heading' and cur.level == level:
                # Found a split point, flush the last section if needed
                if section.first_child is not None:
                    finalizeSection(section)
                    children.append(section)
                    section = Node('MDsection', 0)
            nxt = cur.nxt
            # Avoid adding sections without titles at the start
            if section.first_child is None:
                if cur.t == 'heading' and cur.level == level:
                    section.append_child(cur)
                else:
                    children.append(cur)
            else:
                section.append_child(cur)
            cur = nxt

        # If there's only 1 child then don't bother
        if section.first_child is not None:
            finalizeSection(section)
            children.append(section)

        block.first_child = None
        block.last_child = None
        nextLevel = level + 1
        for child in children:
            # Handle nesting
            if child.t == 'MDsection':
                nestSections(child, level=nextLevel)

            # Append
            if block.first_child is None:
                block.first_child = child
            else:
                block.last_child.nxt = child
            child.parent = block
            child.nxt = None
            child.prev = block.last_child
            block.last_child = child