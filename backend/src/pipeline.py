from sys import path

path.append(r'/src')
path.append(r'../../src')
path.append(r'../')

import tqdm
import copy
from graph2tree import *
import networkx as nx

from pycorenlp import StanfordCoreNLP
from data_utils import convert_to_tree
from pythonds.basic.stack import Stack

warnings.filterwarnings('ignore')

# In[ ]:


_cut_root_node = True
_cut_line_node = True
_cut_pos_node = False
_link_word_nodes = False
_split_sentence = False

source_data_dir = "../data/TextData/"
output_data_dir = "../data/GraphConstruction/"
batch_size = 30
min_freq = 2
max_vocab_size = 15000
seed = 123

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# In[30]:


# test data preprocess
class InputPreprocessor(object):
    def __init__(self, url='http://localhost:9000'):
        self.nlp = StanfordCoreNLP(url)

    def featureExtract(self, src_text, whiteSpace=True):
        if src_text.strip() in preparsed_file.keys():
            return preparsed_file[src_text.strip()]
        print("miss!")
        data = {}
        output = self.nlp.annotate(src_text.strip(), properties={
            'annotators': "tokenize,ssplit,pos,parse",
            "tokenize.options": "splitHyphenated=true,normalizeParentheses=false",
            "tokenize.whitespace": whiteSpace,
            'ssplit.isOneSentence': True,
            'outputFormat': 'json'
        })

        snt = output['sentences'][0]["tokens"]
        depency = output['sentences'][0]["basicDependencies"]
        data["tok"] = []
        data["pos"] = []
        data["dep"] = []
        data["governor"] = []
        data["dependent"] = []
        data['parse'] = output['sentences'][0]['parse']
        for snt_tok in snt:
            data["tok"].append(snt_tok['word'])
            data["pos"].append(snt_tok['pos'])
        for deps in depency:
            data["dep"].append(deps['dep'])
            data["governor"].append(deps['governor'])
            data["dependent"].append(deps['dependent'])
        return data


def get_preparsed_file():
    if not os.path.exists(output_data_dir + "/file_for_parsing.pkl"):
        file_for_parsing = {}
        processor_tmp = InputPreprocessor()

        with open(source_data_dir + "all.txt", "r") as f:
            lines = f.readlines()
            for l in tqdm.tqdm(lines):
                str_ = l.strip().split('\t')[0]
                file_for_parsing[str_] = processor_tmp.featureExtract(str_)

        pkl.dump(file_for_parsing, open(output_data_dir + "./file_for_parsing.pkl", "wb"))
    else:
        preparsed_file = pkl.load(open(output_data_dir + "./file_for_parsing.pkl", "rb"))
    return preparsed_file


preparsed_file = get_preparsed_file()


class Node():
    def __init__(self, word, type_, id_):
        # word: this node's text
        self.word = word

        # type=0 for word nodes, 1 for constituency nodes, 2 for dependency nodes(if they exist)
        self.type = type_

        # id: unique identifier for every node
        self.id = id_

        self.head = False

        self.tail = False

    def __str__(self):
        return self.word


def split_str(string):
    if " . " not in string:
        return [string]
    else:
        s_arr = string.split(" . ")
        res = []
        for s in s_arr:
            if s[-1] != "." and s != s_arr[-1]:
                s = s + " ."
            res.append(s)
        return res


def cut_root_node(con_string):
    tmp = con_string
    if con_string[0] == '(' and con_string[-1] == ')':
        tmp = con_string[1:-1].replace("ROOT", "")
        if tmp[0] == '\n':
            tmp = tmp[1:]
    return tmp


def cut_pos_node(g):
    node_arr = list(g.nodes())
    del_arr = []
    for n in node_arr:
        edge_arr = list(g.edges())
        cnt_in = 0
        cnt_out = 0
        for e in edge_arr:
            if n.id == e[0].id:
                cnt_out += 1
                out_ = e[1]
            if n.id == e[1].id:
                cnt_in += 1
                in_ = e[0]
        if cnt_in == 1 and cnt_out == 1 and out_.type == 0:
            del_arr.append((n, in_, out_))
    for d in del_arr:
        g.remove_node(d[0])
        g.add_edge(d[1], d[2])
    return g


def cut_line_node(g):
    node_arr = list(g.nodes())

    for n in node_arr:
        edge_arr = list(g.edges())
        cnt_in = 0
        cnt_out = 0
        for e in edge_arr:
            if n.id == e[0].id:
                cnt_out += 1
                out_ = e[1]
            if n.id == e[1].id:
                cnt_in += 1
                in_ = e[0]
        if cnt_in == 1 and cnt_out == 1:
            g.remove_node(n)
            #             print "remove", n
            g.add_edge(in_, out_)
    #             print "add_edge", in_, out_
    return g


def get_seq_nodes(g):
    res = []
    node_arr = list(g.nodes())
    for n in node_arr:
        if n.type == 0:
            res.append(copy.deepcopy(n))
    return sorted(res, key=lambda x: x.id)


def get_non_seq_nodes(g):
    res = []
    node_arr = list(g.nodes())
    for n in node_arr:
        if n.type != 0:
            res.append(copy.deepcopy(n))
    return sorted(res, key=lambda x: x.id)


def get_all_text(g):
    seq_arr = get_seq_nodes(g)
    nonseq_arr = get_non_seq_nodes(g)
    seq = [x.word for x in seq_arr]
    nonseq = [x.word for x in nonseq_arr]
    return seq + nonseq


def get_all_id(g):
    seq_arr = get_seq_nodes(g)
    nonseq_arr = get_non_seq_nodes(g)
    seq = [x.id for x in seq_arr]
    nonseq = [x.id for x in nonseq_arr]
    return seq + nonseq


def get_id2word(g):
    res = {}
    seq_arr = get_seq_nodes(g)
    nonseq_arr = get_non_seq_nodes(g)
    for x in seq_arr:
        res[x.id] = x.word
    for x in nonseq_arr:
        res[x.id] = x.word
    return res


def nodes_to_string(l):
    return " ".join([x.word for x in l])


def print_edges(g):
    edge_arr = list(g.edges())
    for e in edge_arr:
        print(e[0].word, e[1].word), (e[0].id, e[1].id)


def print_nodes(g, he_ta=False):
    nodes_arr = list(g.nodes())
    if he_ta:
        print([(n.word, n.id, n.head, n.tail) for n in nodes_arr])
    else:
        print([(n.word, n.id) for n in nodes_arr])


def graph_connect(a_, b_):
    a = copy.deepcopy(a_)
    b = copy.deepcopy(b_)
    max_id = 0
    for n in a.nodes():
        if n.id > max_id:
            max_id = n.id
    tmp = copy.deepcopy(b)
    for n in tmp.nodes():
        n.id += max_id

    res = nx.union(a, tmp)
    seq_nodes_arr = []
    for n in res.nodes():
        if n.type == 0:
            seq_nodes_arr.append(n)
    seq_nodes_arr.sort(key=lambda x: x.id)
    for idx in range(len(seq_nodes_arr)):
        if idx != len(seq_nodes_arr) - 1 and seq_nodes_arr[idx].tail == True:
            if seq_nodes_arr[idx + 1].head == True:
                res.add_edge(seq_nodes_arr[idx], seq_nodes_arr[idx + 1])
                res.add_edge(seq_nodes_arr[idx + 1], seq_nodes_arr[idx])
    return res


def get_vocab(g):
    a = set()
    for n in list(g.nodes()):
        a.add(n.word)
    return a


def get_adj(g):
    # reverse the direction
    adj_dict = {}
    for node, n_dict in g.adjacency():
        adj_dict[node.id] = []

    for node, n_dict in g.adjacency():
        for i in n_dict.items():
            adj_dict[i[0].id].append(node.id)
    return adj_dict


def get_constituency_graph(input_tmp):
    tmp_result = input_tmp

    if _cut_root_node:
        parse_str = cut_root_node(str(tmp_result['parse']))
    else:
        parse_str = str(tmp_result['parse'])
    for punc in ['(', ')']:
        parse_str = parse_str.replace(punc, ' ' + punc + ' ')
    parse_list = str(parse_str).split()

    res_graph = nx.DiGraph()
    pstack = Stack()
    idx = 0
    while idx < len(parse_list):
        if parse_list[idx] == '(':
            new_node = Node(word=parse_list[idx + 1], id_=idx + 1, type_=1)
            res_graph.add_node(new_node)
            pstack.push(new_node)

            if pstack.size() > 1:
                node_2 = pstack.pop()
                node_1 = pstack.pop()
                res_graph.add_edge(node_1, node_2)
                pstack.push(node_1)
                pstack.push(node_2)
        elif parse_list[idx] == ')':
            pstack.pop()
        elif parse_list[idx] in tmp_result['tok']:
            new_node = Node(word=parse_list[idx], id_=idx, type_=0)
            node_1 = pstack.pop()
            if node_1.id != new_node.id:
                res_graph.add_edge(node_1, new_node)
            pstack.push(node_1)
        idx += 1

    max_id = 0
    for n in res_graph.nodes():
        if n.type == 0 and n.id > max_id:
            max_id = n.id

    min_id = 99999
    for n in res_graph.nodes():
        if n.type == 0 and n.id < min_id:
            min_id = n.id

    for n in res_graph.nodes():
        if n.type == 0 and n.id == max_id:
            n.tail = True
        if n.type == 0 and n.id == min_id:
            n.head = True
    return res_graph


def generate_batch_graph(string_batch):
    # generate constituency graph
    graph_list = []
    processor = InputPreprocessor()
    max_node_size = 0
    for s in string_batch:

        # generate multiple graph
        if _split_sentence:
            s_arr = split_str(s)

            g = cut_line_node(get_constituency_graph(processor.featureExtract(s_arr[0])))
            for sub_s in s_arr:
                if sub_s != s_arr[0]:
                    tmp = cut_line_node(get_constituency_graph(processor.featureExtract(sub_s)))
                    g = graph_connect(g, tmp)

        # decide how to cut nodes
        if _cut_pos_node:
            g = cut_pos_node(get_constituency_graph(processor.featureExtract(s)))
        elif _cut_line_node:
            g = cut_line_node(get_constituency_graph(processor.featureExtract(s)))
        else:
            g = (get_constituency_graph(processor.featureExtract(s)))

        if len(list(g.nodes())) > max_node_size:
            max_node_size = len(list(g.nodes()))
        graph_list.append(g)

    info_list = []
    batch_size = len(string_batch)
    for index in range(batch_size):
        word_list = get_all_text(graph_list[index])
        word_len = len(get_seq_nodes(graph_list[index]))
        id_arr = get_all_id(graph_list[index])
        adj_dic = get_adj(graph_list[index])
        new_dic = {}

        # transform id to position in wordlist
        for k in adj_dic.keys():
            new_dic[id_arr.index(k)] = [id_arr.index(x) for x in adj_dic[k]]

        info = {}

        g_ids = {}
        g_ids_features = {}
        g_adj = {}

        for idx in range(max_node_size):
            g_ids[idx] = idx
            if idx < len(word_list):
                g_ids_features[idx] = word_list[idx]

                if _link_word_nodes:
                    if idx <= word_len - 1:
                        if idx == 0:
                            new_dic[idx].append(idx + 1)
                        elif idx == word_len - 1:
                            new_dic[idx].append(idx - 1)
                        else:
                            new_dic[idx].append(idx - 1)
                            new_dic[idx].append(idx + 1)

                g_adj[idx] = new_dic[idx]
            else:
                g_ids_features[idx] = '<P>'
                g_adj[idx] = []

        info['g_ids'] = g_ids
        info['g_ids_features'] = g_ids_features
        info['g_adj'] = g_adj

        info_list.append(info)

        # with open(output_file, "a+") as f:
        return info_list[0]

    batch_vocab = []
    for x in graph_list:
        non_arr = nodes_to_string(get_non_seq_nodes(x)).split()
        for w in non_arr:
            if w not in batch_vocab:
                batch_vocab.append(w)
    return batch_vocab


def test_data_preprocess(line):
    data = []
    managers = pkl.load(open("{}/map.pkl".format(output_data_dir), "rb"))
    word_manager, form_manager = managers
    l_list = line.split("\t")
    w_list = l_list[0].strip().split(' ')
    r_list = form_manager.get_symbol_idx_for_list(l_list[1].strip().split(' '))
    cur_tree = convert_to_tree(r_list, 0, len(r_list), form_manager)
    data.append((w_list, r_list, cur_tree))
    index = 0
    if index != len(data):
        input_batch = [" ".join(data[idx][0]) for idx in range(index, len(data))]
        result = generate_batch_graph(string_batch=input_batch)

    return [data[0], result]


# In[89]:


def cons_batch_graph(graphs):
    g_ids = {}
    g_ids_features = {}
    g_fw_adj = {}
    g_bw_adj = {}
    g_nodes = []
    for g in graphs:
        ids = g['g_ids']
        id_adj = g['g_adj']
        features = g['g_ids_features']
        nodes = []

        # we first add all nodes into batch_graph and create a mapping from graph id to batch_graph id, this mapping will be
        # used in the creation of fw_adj and bw_adj

        id_gid_map = {}
        offset = len(g_ids.keys())
        for id in ids:
            id = int(id)
            g_ids[offset + id] = len(g_ids.keys())
            g_ids_features[offset + id] = features[id]
            id_gid_map[id] = offset + id
            nodes.append(offset + id)
        g_nodes.append(nodes)

        for id in id_adj:
            adj = id_adj[id]
            id = int(id)
            g_id = id_gid_map[id]
            if g_id not in g_fw_adj:
                g_fw_adj[g_id] = []
            for t in adj:
                t = int(t)
                g_t = id_gid_map[t]
                g_fw_adj[g_id].append(g_t)
                if g_t not in g_bw_adj:
                    g_bw_adj[g_t] = []
                g_bw_adj[g_t].append(g_id)

    node_size = len(g_ids.keys())
    for id in range(node_size):
        if id not in g_fw_adj:
            g_fw_adj[id] = []
        if id not in g_bw_adj:
            g_bw_adj[id] = []

    graph = {}
    graph['g_ids'] = g_ids
    graph['g_ids_features'] = g_ids_features
    graph['g_nodes'] = g_nodes
    graph['g_fw_adj'] = g_fw_adj
    graph['g_bw_adj'] = g_bw_adj
    return graph


# In[61]:


from graph2tree import *

warnings.filterwarnings('ignore')

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

checkpoint = torch.load(r'checkpoint_dir/output_model')
encoder = checkpoint["encoder"]
decoder = checkpoint["decoder"]
attention_decoder = checkpoint["attention_decoder"]

encoder.eval()
decoder.eval()
attention_decoder.eval()

managers = pkl.load(open(r"../data/GraphConstruction/map.pkl", "rb"))
word_manager, form_manager = managers

# data = pkl.load(open("{}/test.pkl".format(args.data_dir), "rb"))
# graph_test_list = graph_utils.read_graph_data("{}/graph.test".format(args.data_dir))

reference_list = []
candidate_list = []


# save as pandas dataframe
def g2tree(eqaution, graph_info):
    x = eqaution
    reference = x[1]
    graph_batch = cons_batch_graph([graph_info])
    graph_input = graph_utils.vectorize_batch_graph(graph_batch, word_manager)

    candidate = do_generate(encoder, decoder, attention_decoder, graph_input, word_manager, form_manager, '', True,
                            checkpoint)
    candidate = [int(c) for c in candidate]
    ans = ' '.join([form_manager.idx2symbol[int(c)] for c in candidate])
    # print("Question: ", data[i])
    num_left_paren = sum(1 for c in candidate if form_manager.idx2symbol[int(c)] == "(")
    num_right_paren = sum(1 for c in candidate if form_manager.idx2symbol[int(c)] == ")")
    diff = num_left_paren - num_right_paren

    if diff > 0:
        for i in range(diff):
            candidate.append(form_manager.symbol2idx[")"])
    elif diff < 0:
        candidate = candidate[:diff]
    ref_str = convert_to_string(reference, form_manager)
    cand_str = convert_to_string(candidate, form_manager)

    reference_list.append(reference)
    candidate_list.append(candidate)
    print(ans)
    return ans


# In[98]:


import graph
from comparator_tree import *

# In[110]:


df = pd.read_pickle('../data/TextData/all_df.pkl')

tree = df['tree']


def get_ans(ques):
    try:
        res = test_data_preprocess(ques + '\t' + 'x = 1 * 100')
    except:
        res = test_data_preprocess(
            "there are 2 baskets . there are 1 apples in each basket . how many apples are there in all ?	x = ( 2 * 1 )")
    eqn = g2tree(res[0], res[1])
    print(eqn)
    eqn_parsed = graph.eq_parser(eqn)
    tmp = []
    for j in range(len(tree)):
        if len(tree[j]) == 0:
            continue
        if compare_tree(eqn_parsed[0], tree[j][0]) and compare_tree(eqn_parsed[1], tree[j][1]):
            tmp.append(j)

    return tmp


from flask import Flask, Response, request

app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello World!"


@app.route('/solve', methods=['GET', 'POST'])
def eqn_solver_flask():
    if request.method == 'POST':
        print(request.json['question'])
        return {'list': get_ans(request.json['question'])}
        # return question
    # print(ques)
    return {'list': 'wrong format'}


app.run(host='0.0.0.0', port=8000)
