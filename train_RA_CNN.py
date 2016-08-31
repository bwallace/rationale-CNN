from __future__ import print_function
import math
import csv
import random 
random.seed(1337)
import pickle
import sys
csv.field_size_limit(sys.maxsize)
import os 
import configparser
import optparse 

import sklearn 
from sklearn.metrics import accuracy_score

import pandas as pd 
import numpy as np 

import gensim 
from gensim.models import Word2Vec

from keras.callbacks import ModelCheckpoint


import rationale_CNN
from rationale_CNN import Document


def load_trained_w2v_model(path="/work/03213/bwallace/maverick/RoB_CNNs/PubMed-w2v.bin"):
    m = Word2Vec.load_word2vec_format(path, binary=True)
    return m

def read_data(path="/work/03213/bwallace/maverick/RoB-keras/RoB-data/train-Xy-w-sentences2-Random-sequence-generation.txt"):
    ''' 
    Assumes data is in CSV with following format:

        doc_id,doc_lbl,sentence_number,sentence,sentence_lbl

    Note that we assume sentence_lbl \in {-1, 1}
    '''
    df = pd.read_csv(path)
    # replace empty entries (which were formerly being converted to NaNs)
    # with ""
    df = df.replace(np.nan,' ', regex=True)

    docs = df.groupby("doc_id")
    documents = []
    for doc_id, doc in docs:
        # only need the first because document-level labels are repeated
        doc_label = (doc["doc_lbl"].values[0]+1)/2 # convert to 0/1

        sentences = doc["sentence"].values
        sentence_labels = (doc["sentence_lbl"].values+1)/2
        
        # convert to binary output vectors, so that e.g., [1, 0, 0]
        # indicates a positive rationale; [0, 1, 0] a negative rationale
        # and [0, 0, 1] a non-rationale
        def _to_vec(sent_y): 
            sent_lbl_vec = np.zeros(3)
            if sent_y == 0:
                sent_lbl_vec[-1] = 1.0
            else: 
                # then it's a rationale
                if doc_label > 0: 
                    # positive rationale
                    sent_lbl_vec[0] = 1.0
                else: 
                    # negative rationale
                    sent_lbl_vec[1] = 1.0
            return sent_lbl_vec

        sentence_label_vectors = [_to_vec(sent_y) for sent_y in sentence_labels]
        cur_doc = Document(doc_id, sentences, doc_label, sentence_label_vectors)
        documents.append(cur_doc)

    return documents




def line_search_train(data_path, wvs_path, documents=None, test_mode=False, 
                                model_name="rationale-CNN", 
                                nb_epoch_sentences=20, nb_epoch_doc=25, val_split=.1,
                                sent_dropout_range=(0,.9), num_steps=20,
                                document_dropout=0.5, run_name="RSG",
                                shuffle_data=False, n_filters=32, max_features=20000, 
                                max_sent_len=25, max_doc_len=200,
                                end_to_end_train=False, downsample=False):
    '''
    NOTE: at the moment this is using *all* training data; obviously need to set 
    aside the actual test fold (as we did for the paper experiments in Theano
    implementation). 
    '''

    # read in the docs just once. 
    documents = read_data(path=data_path)
    if shuffle_data: 
        random.shuffle(documents)

    perf_d = {}
    best_so_far = -np.inf
    sent_dropout_star = None
    for sent_dropout in np.linspace(sent_dropout_range[0], sent_dropout_range[1], num_steps):
        r_CNN, documents, p, X_doc, y_doc, best_performance = \
            train_CNN_rationales_model(data_path, wvs_path, documents=documents, 
                                test_mode=test_mode, 
                                model_name=model_name, 
                                nb_epoch_sentences=nb_epoch_sentences, 
                                nb_epoch_doc=nb_epoch_doc, 
                                val_split=val_split,
                                sentence_dropout=float(sent_dropout),
                                document_dropout=document_dropout, 
                                run_name=run_name,
                                shuffle_data=shuffle_data,
                                max_features=max_features, 
                                max_sent_len=max_sent_len, 
                                max_doc_len=max_doc_len,
                                end_to_end_train=end_to_end_train,
                                downsample=downsample)
        
        perf_d[sent_dropout] = best_performance

        print("\n\nbest observed validation performance with sent_dropout_rate: %s was: %s" % (
                    sent_dropout, best_performance))
        if best_performance > best_so_far:
            best_so_far = best_performance
            sent_dropout_star = sent_dropout

    print ("best dropout: %s; best performance: %s" % (sent_dropout_star, best_so_far))

    print("perf-d!")
    print(perf_d)


    with open("perf-d.pickle", "w") as outf:
        pickle.dump(perf_d, outf)



def train_CNN_rationales_model(data_path, wvs_path, documents=None, test_mode=False, 
                                model_name="rationale-CNN", 
                                nb_epoch_sentences=20, nb_epoch_doc=25, val_split=.1,
                                sentence_dropout=0.5, document_dropout=0.5, run_name="RSG",
                                shuffle_data=False, max_features=20000, 
                                max_sent_len=25, max_doc_len=200,
                                n_filters=32,
                                batch_size=50,
                                end_to_end_train=False,
                                downsample=False,
                                stopword=True,
                                pos_class_weight=1):
    
    if documents is None:
        documents = read_data(path=data_path)
        if shuffle_data: 
            random.shuffle(documents)

    wvs = load_trained_w2v_model(path=wvs_path)

    all_sentences = []
    for d in documents: 
        all_sentences.extend(d.sentences)

    p = rationale_CNN.Preprocessor(max_features=max_features, 
                                    max_sent_len=max_sent_len, 
                                    max_doc_len=max_doc_len, 
                                    wvs=wvs, stopword=stopword)

    # need to do this!
    p.preprocess(all_sentences)
    for d in documents: 
        d.generate_sequences(p)

    r_CNN = rationale_CNN.RationaleCNN(p, filters=[3,4,5], 
                                        n_filters=n_filters, 
                                        sent_dropout=sentence_dropout, 
                                        doc_dropout=document_dropout,
                                        end_to_end_train=end_to_end_train)

    ###################################
    # 1. build & train sentence model #
    ###################################
    if model_name == "rationale-CNN":
        r_CNN.build_sentence_model()

        if nb_epoch_sentences > 0:
            print("pre-training sentence model for %s epochs..." % nb_epoch_sentences)
            r_CNN.train_sentence_model(documents, nb_epoch=nb_epoch_sentences, 
                                        sent_val_split=val_split, downsample=True)
            print("done.")


    ###################################
    # 2. build & train document model #
    ###################################
    if model_name == 'doc-CNN':
        print("running **doc_CNN**!")
        r_CNN.build_simple_doc_model()
    else: 
        r_CNN.build_RA_CNN_model()

    '''
    # 8/24
    X_doc, y_doc = [], []
    y_sent = []
    for d in documents:
        cur_X, cur_sent_y = d.get_padded_sequences(p)
        X_doc.append(cur_X)
        y_doc.append(d.doc_y)
        y_sent.append(cur_sent_y)

    X_doc = np.array(X_doc)
    y_doc = np.array(y_doc)
    '''

    # write out model architecture
    json_string = r_CNN.doc_model.to_json() 
    with open("%s_model.json" % model_name, 'w') as outf:
        outf.write(json_string)


    doc_weights_path = "%s_%s.hdf5" % (model_name, run_name)
    # doc_model_path   = "%s_%s_model.h5" % (model_name, run_name)    

    r_CNN.train_document_model(documents, nb_epoch=nb_epoch_doc, 
                                downsample=downsample,
                                batch_size=batch_size,
                                doc_val_split=.2, 
                                pos_class_weight=pos_class_weight,
                                document_model_weights_path=doc_weights_path)
    

    # load best weights back in
    r_CNN.doc_model.load_weights(doc_weights_path)
    # set the final sentence model, which outputs per-sentence
    # predictions regarding rationales. this is admittedly
    # kind of an awkward way of doing things. but in any case
    # now you can call: 
    #   r_CNN.predict_and_rank_sentences_for_doc(new_doc, num_rationales=3) 
    # where new_doc is a Document instance. 
    if model_name == "rationale-CNN":
        r_CNN.set_final_sentence_model()


    # previously, we were using the new .save, which bundles
    # the architecture and weights. however, this is problematic
    # when one goes to load the model due to the use of custom
    # metrics
    # r_CNN.doc_model.save(doc_model_path) # both architecture & weights

    return r_CNN, documents, p



if __name__ == "__main__": 
    parser = optparse.OptionParser()

    parser.add_option('-i', '--inifile',
        action="store", dest="inifile",
        help="path to .ini file", default="config.ini")
    
    parser.add_option('-m', '--model', dest="model",
        help="variant of model to run; one of {rationale_CNN, doc_CNN}", 
        default="rationale-CNN")

    parser.add_option('--se', '--sentence-epochs', dest="sentence_nb_epochs",
        help="number of epochs to (pre-)train sentence model for", 
        default=20, type="int")

    parser.add_option('--de', '--document-epochs', dest="document_nb_epochs",
        help="number of epochs to train the document model for", 
        default=25, type="int")

    parser.add_option('--drops', '--dropout-sentence', dest="dropout_sentence",
        help="sentence-level dropout", 
        default=0.5, type="float")

    parser.add_option('--dropd', '--dropout-document', dest="dropout_document",
        help="document-level dropout", 
        default=0.5, type="float")

    parser.add_option('--val', '--val-split', dest="val_split",
        help="percent of data to hold out for validation", 
        default=0.2, type="float")

    parser.add_option('--n', '--name', dest="run_name",
        help="name of run (e.g., `movies')", 
        default="movies")

    parser.add_option('--tm', '--test-mode', dest="test_mode",
        help="run in test mode?", action='store_true', default=False)

    parser.add_option('--sd', '--shuffle', dest="shuffle_data",
        help="shuffle data?", action='store_true', default=False)

    parser.add_option('--mdl', '--max-doc-length', dest="max_doc_len",
        help="maximum length (in sentences) of a given doc", 
        default=50, type="int")

    parser.add_option('--msl', '--max-sent-length', dest="max_sent_len",
        help="maximum length (in tokens) of a given sentence", 
        default=10, type="int")

    parser.add_option('--mf', '--max-features', dest="max_features",
        help="maximum number of unique tokens", 
        default=20000, type="int")

    parser.add_option('--nf', '--num-filters', dest="n_filters",
        help="number of filters (per n-gram)", 
        default=32, type="int")

    parser.add_option('--pcw', '--pos-class-weight', dest="pos_class_weight",
        help="weight for positive class (relative to neg)", 
        default=1, type="int")

    parser.add_option('--bs', '--batch-size', dest="batch_size",
        help="batch size", 
        default=50, type="int")

    parser.add_option('--tr', '--end-to-end-train', dest="end_to_end_train",
        help="continue training sentence softmax parameters?", 
        action='store_true', default=False)

    parser.add_option('--ls', '--line-search', dest="line_search_sent_dropout",
        help="line search over sentence dropout parameter?", 
        action='store_true', default=False)

    parser.add_option('--ds', '--downsample', dest="downsample",
        help="create balanced mini-batches during training?", 
        action='store_true', default=False)

    parser.add_option('--sw', '--stopword', dest="stopword",
        help="performing stopwording?", 
        action='store_true', default=False)

    (options, args) = parser.parse_args()
  
    config = configparser.ConfigParser()
    print("reading config file: %s" % options.inifile)
    config.read(options.inifile)
    data_path = config['paths']['data_path']
    wv_path   = config['paths']['word_vectors_path']

    print("running model: %s" % options.model)

    if not options.line_search_sent_dropout:
        r_CNN, documents, p = train_CNN_rationales_model(
                                    data_path, wv_path, 
                                    model_name=options.model, 
                                    nb_epoch_sentences=options.sentence_nb_epochs,
                                    nb_epoch_doc=options.document_nb_epochs,
                                    sentence_dropout=options.dropout_sentence, 
                                    document_dropout=options.dropout_document,
                                    run_name=options.run_name,
                                    test_mode=options.test_mode,
                                    val_split=options.val_split,
                                    shuffle_data=options.shuffle_data,
                                    n_filters=options.n_filters,
                                    batch_size=options.batch_size,
                                    max_sent_len=options.max_sent_len,
                                    max_doc_len=options.max_doc_len,
                                    max_features=options.max_features,
                                    end_to_end_train=options.end_to_end_train, 
                                    downsample=options.downsample,
                                    stopword=options.stopword,
                                    pos_class_weight=options.pos_class_weight)
        
        with open("preprocessor.pickle", 'wb') as outf: 
            pickle.dump(p, outf)


        # sanity check!
        doc0 = documents[0]
        pred, rationales = model.predict_and_rank_sentences_for_doc(doc0, num_rationales=2)
        
        #with open("r_CNN.pickle", 'wb') as outf: 
        #    pickle.dump(r_CNN, outf)


    else:
        print("line searching!")
        line_search_train(data_path, wv_path, model_name=options.model, 
                                    nb_epoch_sentences=options.sentence_nb_epochs,
                                    nb_epoch_doc=options.document_nb_epochs,
                                    document_dropout=options.dropout_document,
                                    run_name=options.run_name,
                                    test_mode=options.test_mode,
                                    val_split=options.val_split,
                                    shuffle_data=options.shuffle_data,
                                    n_filters=options.n_filters,
                                    max_sent_len=options.max_sent_len,
                                    max_doc_len=options.max_doc_len,
                                    max_features=options.max_features,
                                    end_to_end_train=options.end_to_end_train,
                                    downsample=options.downsample,
                                    stopword=options.stopword,
                                    pos_class_weight=options.pos_class_weight)
