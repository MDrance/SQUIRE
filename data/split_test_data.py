def create_rel_test_data(dataset:str, relation: str) -> None:
    #Retrieve mappings
    entity2id = {}
    rel2id = {}
    with open(dataset+"/entity2id.txt", "r") as fe2i:
        for line in fe2i:
            e, eid = line.strip().split("\t")
            entity2id[e] = int(eid)
    with open(dataset+"/relation2id.txt", "r") as fr2i:
        for line in fr2i:
            e, eid = line.strip().split("\t")
            rel2id[e] = int(eid)
    #Select test data for a given rel and create the files
    rel_num = len(rel2id)
    rel_test = []
    rel_test_triples = []
    rel_test_triples_rev = []
    with open(dataset+"/test.txt", "r") as fin:
        for triple in fin:
            h,r,t = triple.strip().split("\t")
            if r == relation:
                rel_test.append(str(h)+'\t'+str(r)+'\t'+str(t)+'\n')
                rel_test_triples.append(str(entity2id[h])+'\t'+'R'+str(rel2id[r])+'\t'+str(entity2id[t])+'\n')
                rel_test_triples_rev.append(str(entity2id[h])+'\t'+'R'+str(rel2id[r] + rel_num)+'\t'+str(entity2id[t])+'\n')
    with open(relation + "_test.txt", 'w') as f:
        f.writelines(rel_test)
    with open(relation + "_test_triples.txt", 'w') as f:
        f.writelines(rel_test_triples)
    with open(relation + "_test_triples_rev.txt", 'w') as f:
        f.writelines(rel_test_triples+rel_test_triples_rev)



if __name__ == "__main__":
    create_rel_test_data("OREGANO2", "has_target")