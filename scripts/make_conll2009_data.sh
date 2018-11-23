#!/bin/bash

CONLL09_PATH=$1

SRL_PATH="./data/srl"

if [ ! -d $SRL_PATH ]; then
  mkdir -p $SRL_PATH
fi

EMB_PATH="./embeddings"
if [ ! -d $EMB_PATH ]; then
  mkdir -p $EMB_PATH
fi

ELMO_PATH="./elmo"
if [ ! -d $ELMO_PATH ]; then
  mkdir -p $ELMO_PATH
fi

# Convert CoNLL to json format.
python scripts/conll09_to_json.py "${CONLL09_PATH}/conll09_train.dataset" \
  "${SRL_PATH}/train.english.conll09.jsonlines"

python scripts/conll09_to_json.py "${CONLL09_PATH}/conll09_dev.dataset" \
  "${SRL_PATH}/dev.english.conll09.jsonlines"

python scripts/conll09_to_json.py "${CONLL09_PATH}/conll09_test.dataset" \
  "${SRL_PATH}/test_wsj.english.conll09.jsonlines"

python scripts/conll09_to_json.py "${CONLL09_PATH}/conll09_test_ood.dataset" \
  "${SRL_PATH}/test_brown.english.conll09.jsonlines"

# make gold props

# python scripts/make_conll09_gold_props.py "${CONLL09_PATH}/conll09_dev.dataset" \
#   "${SRL_PATH}/conll09.devel.props.gold.txt"

# python scripts/make_conll09_gold_props.py "${CONLL09_PATH}/conll09_test.dataset" \
#   "${SRL_PATH}/conll09.test.wsj.props.gold.txt"

# python scripts/make_conll09_gold_props.py "${CONLL09_PATH}/conll09_test_ood.dataset" \
#   "${SRL_PATH}/conll09.test.brown.props.gold.txt"


# Filter embeddings.
python scripts/filter_embeddings.py ${EMB_PATH}/glove.840B.300d.txt \
  ${EMB_PATH}/glove.840B.300d.09.filtered \
  ${SRL_PATH}/train.english.conll09.jsonlines ${SRL_PATH}/dev.english.conll09.jsonlines \
  ${SRL_PATH}/test_wsj.english.conll09.jsonlines ${SRL_PATH}/test_brown.english.conll09.jsonlines



# Cache ELMo
# python scripts/cache_elmo.py ${ELMO_PATH}/tfhub.dev_google_elmo_2 ${ELMO_PATH}/conll09.train.elmo_embeddings.hdf5 \
#   ${SRL_PATH}/train.english.conll09.jsonlines

# python scripts/cache_elmo.py ${ELMO_PATH}/tfhub.dev_google_elmo_2 ${ELMO_PATH}/conll09.dev.elmo_embeddings.hdf5 \
#   ${SRL_PATH}/dev.english.conll09.jsonlines

# python scripts/cache_elmo.py ${ELMO_PATH}/tfhub.dev_google_elmo_2 ${ELMO_PATH}/conll09.test_wsj.elmo_embeddings.hdf5 \
#   ${SRL_PATH}/test_wsj.english.conll09.jsonlines

# python scripts/cache_elmo.py ${ELMO_PATH}/tfhub.dev_google_elmo_2 ${ELMO_PATH}/conll09.test_brown.elmo_embeddings.hdf5 \
#   ${SRL_PATH}/test_brown.english.conll09.jsonlines