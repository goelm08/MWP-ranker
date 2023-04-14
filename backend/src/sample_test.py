import pandas as pd

from graph2tree import *

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="parser")
    main_arg_parser.add_argument('-data_dir', type=str, default='../data/GraphConstruction', help='data path')
    main_arg_parser.add_argument('-model', type=str, default='checkpoint_dir/output_model_mawps', help='model checkpoint to use for sampling')
    # main_arg_parser.add_argumentt('-model', type=str, default='../data/GraphConstruction/file_for_parsing.pkl', help='model checkpoint to use for sampling')
    main_arg_parser.add_argument('-seed',type=int,default=123,help='torch manual random number generator seed')

    args = main_arg_parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # checkpoint = torch.load(args.model)
    checkpoint = torch.load(r"C:\Users\goelm\OneDrive\Desktop\model_mawps.pth")
    encoder = checkpoint["encoder"]
    decoder = checkpoint["decoder"]
    attention_decoder = checkpoint["attention_decoder"]

    encoder.eval()
    decoder.eval()
    attention_decoder.eval()

    managers = pkl.load( open("{}/map.pkl".format(args.data_dir), "rb" ) )
    word_manager, form_manager = managers

    data = pkl.load(open("{}/test.pkl".format(args.data_dir), "rb"))
    graph_test_list = graph_utils.read_graph_data("{}/graph.test".format(args.data_dir))

    reference_list = []
    candidate_list = []

    # save as pandas dataframe
    df = pd.DataFrame(columns=['question', 'equation'])

    for i in range(len(data)):
        x = data[i]
        reference = x[1]
        graph_batch = graph_utils.cons_batch_graph([graph_test_list[i]])
        graph_input = graph_utils.vectorize_batch_graph(graph_batch, word_manager)

        candidate = do_generate(encoder, decoder, attention_decoder, graph_input, word_manager, form_manager, args, True, checkpoint)
        candidate = [int(c) for c in candidate]
        ans = ''.join([form_manager.idx2symbol[int(c)] for c in candidate])
        # print("Question: ", data[i])
        question = ' '.join(data[i][0])
        df = df.append({'question': question, 'equation': ans}, ignore_index=True)
        num_left_paren = sum(1 for c in candidate if form_manager.idx2symbol[int(c)]== "(")
        num_right_paren = sum(1 for c in candidate if form_manager.idx2symbol[int(c)]== ")")
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

    #save datafram to file
    df.to_csv('output.csv', index=False)

    print("Test accuracy: ", data_utils.compute_tree_accuracy(candidate_list, reference_list, form_manager))
        


