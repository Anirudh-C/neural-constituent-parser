from nltk.tree import Tree

def phraseCount(tree):
    """
    Counts number of phrases in subtree
    :param: tree
    """
    count = 0
    for child in tree:
        if isinstance(child, Tree):
            count += 1

    return count

def phraseIndices(tree):
    """
    Returns indices of first and second phrase in subtree
    :param: tree
    """
    flag = False
    for i, child in enumerate(tree):
        if isinstance(child, Tree):
            if flag:
                return first, i
            else:
                first = i
                flag = True

def cnf(sentence):
    """
    Returns CNF of phrase tree
    :param: sentence
    """
    node = sentence.label()
    if phraseCount(sentence) > 2:
        # Store current children
        children = [subtree for subtree in sentence]

        # Delete children for given sentence
        sentence[:] = []

        # Compute indices of first and second phrase in children
        first, second = phraseIndices(children)

        # Update sentence
        sentence[:] = children[:second] + [Tree(node, children[second:])]

        # Compute CNF for the next level of phrases
        sentence[first] = cnf(sentence[first])
        sentence[second] = cnf(sentence[second])
    else:
        for i, child in enumerate(sentence):
            if isinstance(child, Tree):
                sentence[i] = cnf(sentence[i])

    return sentence

def collapseUnary(tree):
    """
    Collapse Unary productions in tree if the unary labels are all the same
    :param: tree
    """
    if isinstance(tree, Tree) and len(tree) == 1:
        nodeList = [tree[0]]
    else:
        nodeList = [tree]

    while nodeList:
        node = nodeList.pop()
        if isinstance(node, Tree):
            if len(node) == 1 and \
               isinstance(node[0], Tree) and \
               node.label() == node[0].label():
                node[:] = [child for child in node[0]]
                nodeList.append(node)
            else:
                for child in node:
                    nodeList.append(child)

def collapse(tree):
    """
    Collapses empty labels and paths in tree with the same label
    :param: tree
    """
    for i, child in enumerate(tree):
        if isinstance(child, Tree):
            if not child:
                tree.remove(child)
            else:
                tree[i] = collapse(child)

    collapseUnary(tree)
    return tree

if __name__=="__main__":
    test = Tree("S", ["a", Tree("NP", ["b", Tree("NP", ["c"])]), "d", "e", Tree("VP", ["f"]), "g", "h", Tree("NP", ["i", Tree("NP", ["j"]), Tree("NP", ["k"]), Tree("NP", ["l"])])])
    print("Original Tree:")
    test.draw()
    print("CNF Tree:")
    cnf(test).draw()
    test2 = Tree("S", ["a", Tree("NP", [Tree("NP", ["b"])]), Tree("VP", [Tree("NP", ["c"]), Tree("VP", [Tree("VP", ["d"])])])])
    test2.draw()
    collapse(test2).draw()
