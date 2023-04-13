import pandas as pd


class nptr:
    def __init__(self, c):
        self.data = c
        self.left = None
        self.right = None


def display(node):
    if (node is None):
        print("")
        return
    display(node.left)
    print(node.data, end=" ")
    display(node.right)


def newNode(c):
    n = nptr(c)
    return n


def isDecimal(value):
    try:
        float(value)
        return True
    except ValueError:
        return False



def if_variable(str):
    return str.isalpha()


def is_constant(str):
    return str.isnumeric() or str.isdecimal() or isDecimal(str)


def compare_tree(root1, root2):
    if root1 is None and root2 is None:
        return True

    if root1 is None or root2 is None:
        return False
    # print(root1.data, root2.data)

    if if_variable(root1.data) and if_variable(root2.data):
        return compare_tree(root1.left, root2.left) and compare_tree(root1.right, root2.right)

    if is_constant(root1.data) and is_constant(root2.data):
        return compare_tree(root1.left, root2.left) and compare_tree(root1.right, root2.right)

    if root1.data == root2.data:
        return compare_tree(root1.left, root2.left) and compare_tree(root1.right, root2.right)

    return False


if(__name__ == "__main__"):
    df = pd.read_pickle('data/TextData/test_df.pkl')

    print(df.head())

    tree = df['tree']

    lst = []
    for i in range(len(tree)):
        if len(tree[i]) == 0:
            lst.append([])
            continue
        if i % 100 == 0:
            print(i)
        tmp = []
        for j in range(len(tree)):
            if len(tree[j]) == 0:
                continue
            if compare_tree(tree[i][0], tree[j][0]) and compare_tree(tree[i][1], tree[j][1]):
                tmp.append(j)
        lst.append(tmp)

    df['common'] = lst

    df.to_pickle('data/TextData/test_df_final.pkl')

    print(df.head())
