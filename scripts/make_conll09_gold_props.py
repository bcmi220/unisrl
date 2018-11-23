import sys

def make_gold_props(dataset_path, output_path):
    with open(dataset_path, 'r') as f:
        data = f.readlines()
    
    sentences = []
    sent = []
    for line in data:
        if len(line.strip())>0:
            sent.append(line.strip().split('\t'))
        else:
            if len(sent)>0:
                sentences.append(sent)
                sent = []
    
    if len(sent)>0:
        sentences.append(sent)

    with open(output_path, 'w') as f:
        for sent in sentences:
            for line in sent:
                f.write('\t'.join(line[13:])+'\n')
            f.write('\n')

if __name__ == '__main__':
    make_gold_props(sys.argv[1], sys.argv[2])