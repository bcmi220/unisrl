
import sys

def read_conll(path):
    with open(path, 'r') as f:
        data = f.readlines()

    sentences = []
    sent = []
    for line in data:
        if len(line.strip()) == 0:
            if len(sent)>0:
                sentences.append(sent)
                sent = []
        else:
            sent.append(line.strip().split('\t'))

    if len(sent)>0:
        sentences.append(sent)

    return sentences

def merge_data(gold_data, arg_data):
    assert len(gold_data) == len(arg_data)

    merged_data = []

    for idx in range(len(gold_data)):
        assert len(gold_data[idx]) == len(arg_data[idx])
        new_sent = []
        for jdx in range(len(gold_data[idx])):
            new_sent.append(gold_data[idx][jdx][:12]+[arg_data[idx][jdx][0]]+[gold_data[idx][jdx][13]]+arg_data[idx][jdx][1:])
        merged_data.append(new_sent)

    return merged_data

def save_conll(data, path):
    with open(path, 'w') as f:
        for sent in data:
            for line in sent:
                f.write('\t'.join(line))
                f.write('\n')
            f.write('\n')

if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit("Usage: {} gold_file arg_file out_file [disamb_file]".format(sys.argv[0]))

    gold_data = read_conll(sys.argv[1])

    arg_data = read_conll(sys.argv[2])

    merged_data = merge_data(gold_data, arg_data)

    save_conll(merged_data, sys.argv[3])