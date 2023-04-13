import pandas as pd


def read_text_data_from_file(file_name):
    with open(file_name, 'r') as file:
        text_data = file.read()
    return text_data


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


def build(s):
    stN = []
    stC = []

    p = [0] * 123
    p[ord('+')] = p[ord('-')] = 1
    p[ord('/')] = p[ord('*')] = 2
    p[ord('^')] = 3
    p[ord(')')] = 0

    for ch in s.split():
        # print(i)
        if ch == '(':
            stC.append(ch)
        # Push the operands in node stack
        elif (ch.isalpha() or ch.isnumeric() or ch.isdecimal() or isDecimal(ch)):
            t = newNode(ch)
            stN.append(t)

        elif (p[ord(ch)] > 0):
            while (len(stC) != 0 and stC[-1] != '(' and ((ch != '^' and p[ord(stC[-1])] >= p[ord(ch)])
                                                         or (ch == '^' and
                                                             p[ord(stC[-1])] > p[ord(ch)]))):
                t = newNode(stC[-1])
                stC.pop()
                t1 = stN[-1]
                stN.pop()
                t2 = stN[-1]
                stN.pop()
                t.left = t2
                t.right = t1

                # Push the node to the node stack
                stN.append(t)

            # Push s[i] to char stack
            stC.append(ch)

        elif (ch == ')'):
            while (len(stC) != 0 and stC[-1] != '('):
                t = newNode(stC[-1])
                stC.pop()
                t1 = stN[-1]
                stN.pop()
                # if(len(stN)!=0):
                t2 = stN[-1]
                stN.pop()
                t.left = t2
                t.right = t1
                stN.append(t)
            stC.pop()
    t = stN[-1]
    return t


def eq_parser(eq):
    try:
        eq = eq.split('=')
        eq_lhs = '( ' + eq[0] + ' )'
        eq_rhs = '( ' + eq[1] + ' )'
        print(eq_lhs, eq_rhs)
        return [build(eq_lhs), build(eq_rhs)]
    except:
        print(eq)
        return []


if (__name__ == '__main__'):
    df = pd.read_csv('output_test.csv')

    tree = df['eqn'].apply(eq_parser)

    df['tree'] = tree


    # pandas save dataframe to file
    df.to_pickle('data/TextData/test_df.pkl')

    # pandas read dataframe from file
    df = pd.read_pickle('data/TextData/test_df.pkl')

    print(df.head())