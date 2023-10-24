import os
import json
import random
from random import sample
from tqdm import tqdm
import argparse
import multiprocessing as mp

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FB15K237")
    parser.add_argument("--out", default="6_rev")
    parser.add_argument("--max-len", default=3, type=int)
    parser.add_argument("--num", default=6, type=int)
    parser.add_argument("--rule", default=False, action="store_true") # combine generated rules to sample path
    parser.add_argument("--gen-mapping", default=False, action="store_true") # generate mapping files: entity2id, relation2id
    parser.add_argument("--gen-eval-data", default=False, action="store_true") # generate files for evaluation
    parser.add_argument("--gen-train-data", default=False, action="store_true") # generate files for train (sample path)
    parser.add_argument("--cpu", default=4, type=int) # Nbr of CPU to use to speed up gneration of in and out file
    parser.add_argument("--rel_type", default="", type=int) #Relation to use if we want to only sample paths for this relation
    args = parser.parse_args()
    return args

def read_triple(file_path: str, entity2id: dict, relation2id: dict)-> list:
    '''
    Read triples and map them into ids in the form of mapped triples [(18, 4, 19), (69, 5, 420)]
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def read_mapping(file_path: str, entities: dict, relations: dict) -> None:
    """
    Read a file containing train, val or test triple in textual form.
    Update the dicts by listing each unique node and relation (could have used a set?)

    Parameters
    ----------
    entities, relations : dict
        Dict acting as set to track unique nodes and relations.
    """
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            if h not in entities:
                entities[h] = 1
            if t not in entities:
                entities[t] = 1
            if r not in relations:
                relations[r] = 1

def gen_mapping(args):
    """
    Read the dict of unique nodes and relation create by read_mapping() and create the two vocabs
    """
    print("-------generating mapping files-------")
    entities = {}
    relations = {}
    read_mapping(os.path.join(args.dataset, 'train.txt'), entities, relations)
    read_mapping(os.path.join(args.dataset, 'valid.txt'), entities, relations)
    read_mapping(os.path.join(args.dataset, 'test.txt'), entities, relations)
    ent_lines = []
    num = 0
    for ent in entities.keys():
        ent_lines.append(ent+'\t'+str(num)+'\n')
        num += 1
    rel_lines = []
    num = 0
    for rel in relations.keys():
        rel_lines.append(rel+'\t'+str(num)+'\n')
        num += 1
    with open(os.path.join(args.dataset, 'entity2id.txt'), 'w') as f:
        f.writelines(ent_lines)
    with open(os.path.join(args.dataset, 'relation2id.txt'), 'w') as f:
        f.writelines(rel_lines)
    print("-------finish mapping files-------")

def gen_eval(train_triples: list, valid_triples: list, test_triples: list, relation2id: dict):
    """
    Generate the train, val and test file with mapped triple and their inverses.
    R symbol is added to relation ID.
    Inverse relation are modeled by r_id+nbr_rel
    train_triples, val_triples and test triples contain mapped triples.
    Inverse files contain base triples + inverse.

    Parameters
    ----------
    X_triples : list
        triples in form of mapped triples [(), (),...]
    relation2id : dict
        mapping of relation to their ID
    """
    print("-------generating evaluation files-------")
    rel_num = len(relation2id)
    # valid and test file
    valid_line = []
    valid_line_rev = []
    for valid_triple in valid_triples:
        h, r, t = valid_triple
        valid_line.append(str(h)+'\t'+'R'+str(r)+'\t'+str(t)+'\n')
        valid_line_rev.append(str(t)+'\t'+'R'+str(r+rel_num)+'\t'+str(h)+'\n')
    test_line = []
    test_line_rev = []
    for test_triple in test_triples:
        h, r, t = test_triple
        test_line.append(str(h)+'\t'+'R'+str(r)+'\t'+str(t)+'\n')
        test_line_rev.append(str(t)+'\t'+'R'+str(r+rel_num)+'\t'+str(h)+'\n')
    with open(os.path.join(args.dataset, 'valid_triples.txt'), 'w') as f:
        f.writelines(valid_line)
    with open(os.path.join(args.dataset, 'valid_triples_rev.txt'), 'w') as f:
        f.writelines(valid_line+valid_line_rev)
    with open(os.path.join(args.dataset, 'test_triples.txt'), 'w') as f:
        f.writelines(test_line)
    with open(os.path.join(args.dataset, 'test_triples_rev.txt'), 'w') as f:
        f.writelines(test_line+test_line_rev)
    # filter file
    train_line = []
    train_line_rev = []
    train_triples = train_triples
    for triple in train_triples:
        h, r, t = triple
        train_line.append(str(h)+'\t'+'R'+str(r)+'\t'+str(t)+'\n')
        train_line_rev.append(str(t)+'\t'+'R'+str(r+rel_num)+'\t'+str(h)+'\n')
    with open(os.path.join(args.dataset, 'train_triples.txt'), 'w') as f:
        f.writelines(train_line)
    with open(os.path.join(args.dataset, 'train_triples_rev.txt'), 'w') as f:
        f.writelines(train_line+train_line_rev)
    print("-------finish evaluation files-------")

def gen_path_from_rule(adjacent: dict, rel2rules: dict, triple: list, num: int):
    """
    Generate paths between h and t using the mined rules. 
    Return a list with 6 possible paths from h to t.

    Parameters
    ----------
    adjacent : dict
        keep track of nodes linked to h by a relation
            {start: {edge1: [end11, end12, ...]}, edge2: [end21, end22, ...], ...}, ...}
    rel2rules : dict
        mapping of head relation and body relations and confidence
            {18 : [[0.51, [45, 56, 23]], [0.27, [89, 25, 01]]], 95 : [[0.36, [82, 51, 23]], 0.66, [78, 10, 09]]]}
    triple : list
        Mapped triples
    """
    total = 0
    start = triple[0]
    end = triple[2]
    rel = str(triple[1])
    if rel in rel2rules:
        rules = rel2rules[rel]
    else:
        return 0, []
    rule_paths = []
    for meta_rule in rules:
        paths = []
        new_candidate_paths = [[start]]
        rule = meta_rule[1]
        l = len(rule)
        for i in range(l):
            edge = rule[i]
            candidate_paths = new_candidate_paths
            new_candidate_paths = []
            for candidate_path in candidate_paths:
                if len(paths) > 10:
                    break
                pre = candidate_path[-1]
                sucs = []
                if pre in adjacent and edge in adjacent[pre]:
                    sucs = adjacent[pre][edge]
                for suc in sucs:
                    if i < l - 1:
                        new_candidate_paths.append(candidate_path + [edge, suc])
                    elif suc == end:
                        paths.append(candidate_path + [edge, suc])
        if len(paths) > 0:
            rule_paths.append(sample(paths, 1)[0])
            total += 1
            if total == num:
                break
    return total, rule_paths

def sample_path(len2paths: list, num: int):
    sample_paths = []
    N = 0
    for paths in len2paths:
        if len(paths) > 0:
            N += 1
    for paths in len2paths:
        l = len(paths)
        if l > 0:
            sample_per_len = int(num / N)
            num -= sample_per_len
            N -= 1
            if l > sample_per_len:
                sample_paths += sample(paths, sample_per_len)
            else:
                repeat = int(sample_per_len / l)
                rest = sample_per_len - repeat * l
                for i in range(repeat):
                    sample_paths += paths
                sample_paths += sample(paths, rest)
    return sample_paths

def inv(edge, rel_num, rev):
    if not rev:
        return edge
    elif edge >= rel_num:
        return (edge - rel_num)
    else:
        return (edge + rel_num)

def write_path(start, edge, paths, in_line, out_line, rel_num, rev):
    edge = inv(edge, rel_num, rev)
    for path in paths:
        if rev:
            path = path[::-1]
        in_line.append(str(start)+' '+'R'+str(edge)+'\n')
        l = int(len(path)/2)+1
        line = ""
        if l == 1: # take care of self-loop
            line += ('R'+str(edge)+' '+str(end)+'\n')
        for i in range(1, l):
            if i < l - 1:
                line += ('R'+str(inv(path[2*i-1], rel_num, rev))+' '+str(path[2*i])+' ')
            else:
                line += ('R'+str(inv(path[2*i-1], rel_num, rev))+' '+str(path[2*i])+'\n')
        out_line.append(line)

def find_path(connected, start, rel, end, max_len=3):
    if rel != args.rel_type:
        return [[]]
    paths = []
    if start not in connected:
        return [[]]
    # one-hop
    path1 = []
    if end in connected[start]:
        for label in connected[start][end]:
            path1.append([start, label, end])
    paths.append(path1)
    if max_len == 1:
        return paths
    # two-hop
    path2 = []
    for mid in connected[start]:
        if mid == end or mid == start:
            continue
        if end not in connected[mid]:
            continue
        labels1 = connected[start][mid]
        labels2 = connected[mid][end]
        for label2 in labels2:
            for label1 in labels1:
                path2.append([start, label1, mid, label2, end])
    paths.append(path2)
    if max_len == 2:
        return paths
    # three-hop
    path3 = []
    for mid1 in connected[start]:
        if mid1 == end or mid1 == start:
            continue
        for mid2 in connected[mid1]:
            if mid2 == end or mid2 == start or mid2 == mid1:
                continue
            if end not in connected[mid2]:
                continue
            labels1 = connected[start][mid1]
            labels2 = connected[mid1][mid2]
            labels3 = connected[mid2][end]
            for label3 in labels3:
                for label2 in labels2:
                    for label1 in labels1:
                        path3.append([start, label1, mid1, label2, mid2, label3, end])
    paths.append(path3)
    if max_len == 3:
        return paths
    # four-hop
    path4 = []
    for mid1 in connected[start]:
        if mid1 == end or mid1 == start:
            continue
        for mid2 in connected[mid1]:
            if mid2 == end or mid2 == start or mid2 == mid1:
                continue
            for mid3 in connected[mid2]:
                if mid3 == end or mid3 == start or mid3 == mid1 or mid3 == mid2:
                    continue
                if end not in connected[mid3]:
                    continue
                labels1 = connected[start][mid1]
                labels2 = connected[mid1][mid2]
                labels3 = connected[mid2][mid3]
                labels4 = connected[mid3][end]
                for label4 in labels4:
                    for label3 in labels3:
                        for label2 in labels2:
                            for label1 in labels1:
                                path4.append([start, label1, mid1, label2, mid2, label3, mid3, label4, end])
    paths.append(path4)
    return paths

def search_path(connected, start, end, max_len):
    candidate_paths = [[start, ]]
    paths = [[] for i in range(max_len+1)]
    while len(candidate_paths):
        path = candidate_paths.pop(0)
        pre = path[-1]
        l = int(len(path)/2)
        if pre == end: # find path
            paths[l].append(path)
            continue
        if l == max_len - 1:
            if pre in connected and end in connected[pre]:
                for edge in connected[pre][end]:
                    paths[max_len].append(path + [edge, end])
        else:
            if pre in connected:
                for suc in connected[pre].keys():
                    if suc not in path[::2]:
                        for edge in connected[pre][suc]:
                            candidate_paths.append(path + [edge, suc])
    return paths

def gen_train(train_triples, relation2id, q_in, q_out, args):
    """
    Generate in_ and out_ files.
    in_file correspond to the request (h, r)
    out_file correspond to the n possible paths from h to t that implies r

    Parameters
    ----------
    train_triples : list
        triples in form of mapped triples [(), (),...]
    relation2id : dict
        mapping between rel and their ID
    """
    print("-------generating training files-------")
    rel_num = len(relation2id)
    all_true_triples = train_triples
    all_reverse_triples = []
    connected = dict() # {start: {end1: [edge11, edge12, ...], end2: [edge21, edge22, ...], ...}, ...}, keep track of relation between two nodes
    adjacent = dict() # {start: {edge1: [end11, end12, ...]}, edge2: [end21, end22, ...], ...}, ...}, keep track of nodes linked to h by a relation
    for triple in all_true_triples:
        all_reverse_triples.append((triple[2], triple[1] + rel_num, triple[0]))
    all_triples = all_reverse_triples + all_true_triples
    for triple in all_triples:
        start = triple[0]
        end = triple[2]
        edge = triple[1]
        if start not in connected:
            connected[start] = dict()
        if start not in adjacent:
            adjacent[start] = dict()
        if end not in connected[start]:
            connected[start][end] = []
        if edge not in adjacent[start]:
            adjacent[start][edge] = []
        connected[start][end].append(edge)
        adjacent[start][edge].append(end)

    in_line = []
    out_line = []
    if args.rule:
        with open(os.path.join(args.dataset, 'rules.dict')) as f:
            rel2rules = json.load(f)
    for triple in tqdm(all_true_triples):
        num = 0
        start = triple[0]
        end = triple[2]
        edge = triple[1]
        paths = list()
        if args.rule:
            adjacent[start][edge].remove(end)
            num, paths = gen_path_from_rule(adjacent, rel2rules, triple, args.num)
            adjacent[start][edge].append(end)
        num = args.num - num
        # find all paths of max_len in remaining graph if not enough paths were found using mined rules
        if num > 0:
            connected[start][end].remove(edge)
            # len2paths = search_path(connected, start, end, args.max_len)
            len2paths = find_path(connected, start, edge, end, args.max_len)
            connected[start][end].append(edge)
            # print(len2paths)
            N = 0
            #print(len2paths)
            for pathss in len2paths:
                if len(pathss) > 0:
                    N += 1
            if N == 0: # no path
                # print("no")
                for n in range(num):
                    in_line.append(str(start)+' '+'R'+str(edge)+'\n')
                    out_line.append('R'+str(edge)+' '+str(end)+'\n')
                for n in range(num):
                    in_line.append(str(end)+' '+'R'+str(edge+rel_num)+'\n')
                    out_line.append('R'+str(edge+rel_num)+' '+str(start)+'\n')
            paths1 = paths + sample_path(len2paths, num)
            paths2 = paths + sample_path(len2paths, num)
        else:
            paths1 = paths
            paths2 = paths
        #print(paths1)
        write_path(start, edge, paths1, in_line, out_line, rel_num, rev=False)
        write_path(end, edge, paths2, in_line, out_line, rel_num, rev=True)
    q_in.put(in_line)
    q_out.put(out_line)

def listener(q_in, q_out, args):
    with open(os.path.join(args.dataset, 'in_'+args.out+'.txt'), 'w') as f:
        while True:
            m = q_in.get()
            if m == "#done#":
                break
            for i in m:
                f.write(i)
            f.flush()
    with open(os.path.join(args.dataset, 'out_'+args.out+'.txt'), 'w') as f:
        while True:
            m = q_out.get()
            if m == "#done#":
                break
            for i in m:
                f.write(i)
            f.flush()

def gen_train_parallel_with_IO(triples, relation2id, num_processes, args):
    manager = mp.Manager()
    q_in = manager.Queue()
    q_out = manager.Queue()
    pool = mp.Pool(num_processes+1)
    watcher = pool.apply_async(listener, (q_in, q_out, args))

    chunk_size = len(triples) // num_processes
    results = []
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size if i < num_processes - 1 else None
        chunk = triples[start:end]

        result = pool.apply_async(gen_train, (chunk, relation2id, q_in, q_out, args))
        results.append(result)

    # Wait for all processes to complete
    for result in results:
        result.get()

    q_in.put("#done#")
    q_out.put("#done#")
    pool.close()
    pool.join()

if __name__ == "__main__":
    random.seed(12345)
    args = get_args()
    lines = []
    entity2id = {}
    relation2id = {}
    if args.gen_mapping:
        gen_mapping(args)
    with open(os.path.join(args.dataset, 'entity2id.txt')) as fin:
        for line in fin:
            e, eid = line.strip().split('\t')
            entity2id[e] = int(eid)

    with open(os.path.join(args.dataset, 'relation2id.txt')) as fin:
        for line in fin:
            r, rid = line.strip().split('\t')
            relation2id[r] = int(rid)

    train_triples = read_triple(os.path.join(args.dataset, 'train.txt'), entity2id, relation2id)
    valid_triples = read_triple(os.path.join(args.dataset, 'valid.txt'), entity2id, relation2id)
    test_triples = read_triple(os.path.join(args.dataset, 'test.txt'), entity2id, relation2id)
    # generate eval data files
    if args.gen_eval_data:
        gen_eval(train_triples, valid_triples, test_triples, relation2id)
    # generate train data files
    if args.gen_train_data:
        # gen_train(train_triples, relation2id, args)
        gen_train_parallel_with_IO(train_triples, relation2id, args.cpu, args)
