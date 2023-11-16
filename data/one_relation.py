import multiprocessing as mp
from tqdm import tqdm
from multiprocessing import Queue

def work_with_one_relation(chunk_in: list, chunk_out: list, 
                           rel: int, inv_rel: int, 
                           q_in: Queue, q_out: Queue, 
                           repeat: int):
    chunk_in = list(set(chunk_in))
    chunk_out = list(set(chunk_out))
    chunk_in_new = []
    chunk_out_new = []
    rels = ["R"+str(rel), "R"+str(inv_rel)]
    for i, j in tqdm(zip(chunk_in, chunk_out), total=len(chunk_in)):
        if i.split()[-1] in rels:
            chunk_in_new.append(i)
            chunk_out_new.append(j)
    chunk_in_new = [ele for ele in chunk_in_new for i in range(repeat)]
    chunk_out_new = [ele for ele in chunk_out_new for i in range(repeat)]
    q_in.put(chunk_in_new)
    q_out.put(chunk_out_new)


def listener(dataset: str, q_in: Queue, q_out: Queue):
    with open(dataset + "/in_6_rev_rule_test.txt", "w") as f:
        while True:
            m = q_in.get()
            if m == "#done#":
                break
            for i in m:
                f.write(i)
            f.flush()
    with open(dataset + "/out_6_rev_rule_test.txt", "w") as f:
        while True:
            m = q_out.get()
            if m == "#done#":
                break
            for i in m:
                f.write(i)
            f.flush()

def work_with_one_relation_parallel(dataset: str, 
                                    rel: int, inv_rel: int, 
                                    num_processes: int,
                                    repeat: int,):
    manager = mp.Manager()
    q_in = manager.Queue()
    q_out = manager.Queue()
    pool = mp.Pool(num_processes+1)
    watcher = pool.apply_async(listener, (dataset, q_in, q_out))
    in_path = dataset + "/in_6_rev_rule.txt"
    out_path = dataset + "/out_6_rev_rule.txt"
    with open(in_path, "r") as fin:
        raw_in = fin.readlines()
    with open(out_path, "r") as fin:
        raw_out =  fin.readlines()
    
    chunk_size = len(raw_in) // num_processes
    results = []
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size if i < num_processes - 1 else None
        chunk_in = raw_in[start:end]
        chunk_out = raw_out[start:end]

        result = pool.apply_async(work_with_one_relation, (chunk_in, 
                                                           chunk_out, 
                                                           rel, 
                                                           inv_rel, 
                                                           q_in, 
                                                           q_out,
                                                           repeat))
        results.append(result)

    # Wait for all processes to complete
    for result in results:
        result.get()

    q_in.put("#done#")
    q_out.put("#done#")
    pool.close()
    pool.join()


if __name__ == "__main__":
    work_with_one_relation_parallel("BioKG/", 2, 15, 100, 6)
