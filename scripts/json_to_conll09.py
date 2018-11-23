import sys
import json

def json_to_conll09(json_path, dataset_path, output_path):
    with open(json_path, 'r') as f:
        data = f.readlines()
    json_data = [json.loads(item.strip()) for item in data if len(item.strip())>0]

    with open(dataset_path, 'r') as f:
        data = f.readlines()

    sentence_data = []
    sent = []
    for line in data:
        if len(line.strip())>0:
            sent.append(line.strip().split('\t'))
        else:
            if len(sent)>0:
                sentence_data.append(sent)
                sent = []
    
    if len(sent)>0:
        sentence_data.append(sent)
        sent = []

    assert len(json_data) == len(sentence_data)

    with open(output_path, 'w') as f:
        for idx in range(len(sentence_data)):
            
            assert 'S'+str(idx) == json_data[idx]['doc_key']

            txt = json_data[idx]['sentences'][0]

            assert len(txt) == len(sentence_data[idx])

            if len(json_data[idx]['predicted_srl']) > 0:
                pred_srl = json_data[idx]['predicted_srl']
                predicates = list(set([item[0] for item in pred_srl]))
                predicates.sort() #
                pred_dict = {pred:idx for idx,pred in enumerate(predicates)}

                sent = sentence_data[idx]
                sent_data = []
                for line in sent:
                    sent_data.append(line[:12]+['_','_']+['_' for _ in range(len(predicates))])

                for prd in predicates:
                    sent_data[prd][12] = 'Y'
                    sent_data[prd][13] = sent[prd][13]
                
                for item in pred_srl:
                    sent_data[item[1]][14+pred_dict[item[0]]] = item[3]

                for line in sent_data:
                    f.write('\t'.join(line)+'\n')
                f.write('\n')
            else:
                sent = sentence_data[idx]
                sent_data = []
                for line in sent:
                    sent_data.append(line[:12]+['_','_'])
                for line in sent_data:
                    f.write('\t'.join(line)+'\n')
                f.write('\n')


if __name__ == '__main__':
    json_to_conll09(sys.argv[1], sys.argv[2], sys.argv[3])