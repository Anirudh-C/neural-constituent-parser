from nltk.tree import Tree
import numpy as np

def cnf(sentence):
    """
    Returns CNF of phrase tree
    :param: sentence
    """
    if isinstance(sentence, Tree):
        node = sentence.label()
        if len(sentence) > 2:
            # Store current children
            children = [subtree for subtree in sentence]

            # Delete children for given sentence
            sentence[:] = []

            # Update sentence
            sentence[:] = children[:1] + [Tree(node, children[1:])]

            # Compute CNF for the next level of phrases
            sentence[0] = cnf(sentence[0])
            sentence[1] = cnf(sentence[1])
        else:
            for i, child in enumerate(sentence):
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

    return tree

def collapseEmpty(tree):
    """
    Collapses empty labels in tree
    :param: tree
    """
    for i, child in enumerate(tree):
        if isinstance(child, Tree):
            if not child:
                tree.remove(child)
            else:
                tree[i] = collapseEmpty(child)

    return tree

def collapse(tree, count=4):
    """
    Combines all the collapse operations
    :param: tree
    :param: count - number of times to run collapse due to
    the operations of unary collapse and empty collapse being
    linked
    """
    if count:
        return collapse(collapseUnary(collapseEmpty(tree)), count-1)
    return collapseEmpty(collapseUnary(tree))

def spanRepresentation(wordVectors, span):
    """
    Given the word vectors for a sentence and a span
    return the span representation
    """
    if len(span) == 4:
        i,j,k,l = span
    else:
        raise ValueError
    if i:
        firstSpan = np.concatenate((np.split(wordVectors[j], 2)[0] - np.split(wordVectors[i-1], 2)[0],
                                    np.split(wordVectors[j+1], 2)[1] - np.split(wordVectors[i], 2)[1]))
    else:
        firstSpan = np.concatenate((np.split(wordVectors[j], 2)[0],
                                    np.split(wordVectors[j+1], 2)[1] - np.split(wordVectors[i], 2)[1]))

    if l == len(wordVectors):
        secondSpan = np.concatenate((np.split(wordVectors[l], 2)[0] - np.split(wordVectors[k-1], 2)[0],
                                     - np.split(wordVectors[k], 2)[1]))
    else:
        secondSpan = np.concatenate((np.split(wordVectors[l], 2)[0] - np.split(wordVectors[k-1], 2)[0],
                                     np.split(wordVectors[l+1], 2)[1] - np.split(wordVectors[k], 2)[1]))

    return np.concatenate((firstSpan, secondSpan))

if __name__=="__main__":
    test = Tree("S", ["a", Tree("NP", ["b", Tree("NP", ["c"])]), "d", "e", Tree("VP", ["f"]), "g", "h", Tree("NP", ["i", Tree("NP", ["j"]), Tree("NP", ["k"]), Tree("NP", ["l"])])])
    print("Original Tree:")
    test.draw()
    print("CNF Tree:")
    cnf(test).draw()
    
