'''
@authors Byron Wallace, Edward Banner, Ye Zhang, Iain Marshall

A Keras implementation of our "rationale augmented CNN" (https://arxiv.org/abs/1605.04469). Please note that
the model was originally implemented in Theano; results reported in the paper are from said implementation. 
This version is a work in progress. 

Credit for initial pass of basic CNN implementation to: Cheng Guo (https://gist.github.com/entron).

References
--
Ye Zhang, Iain J. Marshall and Byron C. Wallace. "Rationale-Augmented Convolutional Neural Networks for Text Classification". http://arxiv.org/abs/1605.04469
Yoon Kim. "Convolutional Neural Networks for Sentence Classification". EMNLP 2014.
Ye Zhang and Byron Wallace. "A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification". http://arxiv.org/abs/1510.03820.
& c.f. http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
'''

from __future__ import print_function
import pdb
import sys
try:
    reload(sys)
    sys.setdefaultencoding('utf8')
except:
    # almost certainly means Python 3x
    pass 

import random


import numpy as np

from keras.optimizers import SGD, RMSprop
from keras import backend as K 
from keras.models import Graph, Model, Sequential
from keras.preprocessing import sequence
from keras.engine.topology import Layer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Dense, merge
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge, Reshape, Permute, Lambda
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D
from keras.utils.np_utils import accuracy
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.constraints import maxnorm

class RationaleCNN:

    def __init__(self, preprocessor, filters=None, n_filters=20,#32, 
                        sent_dropout=0.5, doc_dropout=0.5, 
                        end_to_end_train=False):
        '''
        parameters
        ---
        preprocessor: an instance of the Preprocessor class, defined below
        '''
        self.preprocessor = preprocessor

        if filters is None:
            self.ngram_filters = [3, 4, 5]
        else:
            self.ngram_filters = filters 

      
        #self.ngram_filters = [1, 2]
        self.n_filters = n_filters 
        self.sent_dropout = sent_dropout
        self.doc_dropout  = doc_dropout
        self.sentence_model_trained = False 
        self.end_to_end_train = end_to_end_train

    @staticmethod
    def get_weighted_sum_func(X, weights):
        # @TODO.. add sentence preds!
        def weighted_sum(X):
            return K.sum(np.multiply(X, weights), axis=-1)
        
        #return K.sum(X, axis=0) 
        return weighted_sum

    @staticmethod
    def weighted_sum_output_shape(input_shape):
        # expects something like (None, max_doc_len, num_features) 
        # returns (1 x num_features)
        shape = list(input_shape)
        return tuple((1, shape[-1]))

    @staticmethod
    def balanced_sample(X, y):
        _, pos_rationale_indices = np.where([y[:,0] > 0]) 
        _, neg_rationale_indices = np.where([y[:,1] > 0]) 
        _, non_rationale_indices = np.where([y[:,2] > 0]) 

        # sample a number of non-rationales equal to the total
        # number of pos/neg rationales. 
        m = pos_rationale_indices.shape[0] + neg_rationale_indices.shape[0]
                                        # np.array(random.sample(non_rationale_indices, m)) 
        sampled_non_rationale_indices = np.random.choice(non_rationale_indices, m, replace=False)

        train_indices = np.concatenate([pos_rationale_indices, neg_rationale_indices, sampled_non_rationale_indices])
        np.random.shuffle(train_indices) # why not
        return X[train_indices,:], y[train_indices]


    def build_simple_doc_model(self):
        # maintains sentence structure, but does not impose weights.
        tokens_input = Input(name='input', 
                            shape=(self.preprocessor.max_doc_len, self.preprocessor.max_sent_len), 
                            dtype='int32')

        tokens_reshaped = Reshape([self.preprocessor.max_doc_len*self.preprocessor.max_sent_len])(tokens_input)

    
        x = Embedding(self.preprocessor.max_features, self.preprocessor.embedding_dims, 
                        weights=self.preprocessor.init_vectors,
                        name="embedding")(tokens_reshaped)

        x = Reshape((1, self.preprocessor.max_doc_len, 
                     self.preprocessor.max_sent_len*self.preprocessor.embedding_dims), 
                     name="reshape")(x)

        convolutions = []

        for n_gram in self.ngram_filters:
            cur_conv = Convolution2D(self.n_filters, 1, 
                                     n_gram*self.preprocessor.embedding_dims, 
                                     subsample=(1, self.preprocessor.embedding_dims),
                                     activation="relu",
                                     name="conv2d_"+str(n_gram))(x)

            # this output (n_filters x max_doc_len x 1)
            one_max = MaxPooling2D(pool_size=(1, self.preprocessor.max_sent_len - n_gram + 1), 
                                   name="pooling_"+str(n_gram))(cur_conv)

            # flip around, to get (1 x max_doc_len x n_filters)
            permuted = Permute((2,1,3), name="permuted_"+str(n_gram)) (one_max)
            
            # drop extra dimension
            r = Reshape((self.preprocessor.max_doc_len, self.n_filters), 
                            name="conv_"+str(n_gram))(permuted)
            
            convolutions.append(r)


        sent_vectors = merge(convolutions, name="sentence_vectors", mode="concat")
        sent_vectors = Dropout(self.sent_dropout, name="dropout")(sent_vectors)

        '''
        For this model, we simply take an unweighted sum of the sentence vectors
        to induce a document representation.
        ''' 
        def sum_sentence_vectors(x):
            return K.sum(x, axis=1)

        def sum_sentence_vector_output_shape(input_shape): 
            # should be (batch x max_doc_len x sentence_dim)
            shape = list(input_shape) 
            # something like (None, 96), where 96 is the
            # length of induced sentence vectors
            return (shape[0], shape[-1])
            
        doc_vector = Lambda(sum_sentence_vectors, 
                                output_shape=sum_sentence_vector_output_shape,
                                name="document_vector")(sent_vectors)

        doc_vector = Dropout(self.doc_dropout, name="doc_v_dropout")(doc_vector)
        output = Dense(1, activation="sigmoid", name="doc_prediction")(doc_vector)

        self.doc_model = Model(input=tokens_input, output=output)

        self.doc_model.compile(metrics=["accuracy"], loss="binary_crossentropy", optimizer="adadelta")
        print("doc-CNN model summary:")
        print(self.doc_model.summary())


    def build_RA_CNN_model(self):
        assert self.sentence_model_trained

        # input dim is (max_doc_len x max_sent_len) -- eliding the batch size
        tokens_input = Input(name='input', 
                            shape=(self.preprocessor.max_doc_len, self.preprocessor.max_sent_len), 
                            dtype='int32')
        
        # flatten; create a very wide matrix to hand to embedding layer
        tokens_reshaped = Reshape([self.preprocessor.max_doc_len*self.preprocessor.max_sent_len])(tokens_input)
        # embed the tokens; output will be (p.max_doc_len*p.max_sent_len x embedding_dims)
        # here we should initialize with weights from sentence model embedding layer!
        # also pass weights for initialization
        x = Embedding(self.preprocessor.max_features, self.preprocessor.embedding_dims, 
                        weights=self.sentence_model.get_layer("embedding").get_weights(),
                        #weights=self.preprocessor.init_vectors, 
                        name="embedding")(tokens_reshaped)

        # reshape to preserve document structure; each doc will now be a
        # a row in this matrix
        x = Reshape((1, self.preprocessor.max_doc_len, 
                     self.preprocessor.max_sent_len*self.preprocessor.embedding_dims), 
                     name="reshape")(x)

        
        total_sentence_dims = len(self.ngram_filters) * self.n_filters 

        convolutions = []
        for n_gram in self.ngram_filters:
            layer_name = "conv_" + str(n_gram)
            # pull out the weights from the sentence model.
            cur_conv_layer = self.sentence_model.get_layer(layer_name)
            weights, biases = cur_conv_layer.get_weights()
            # here it gets a bit tricky; we need dims 
            #       (nb_filters x 1 x 1 x (n_gram*embedding_dim))
            # for 2d conv; our 1d conv model, though, will have
            #       (nb_filters x embedding_dim x n_gram x 1)
            # need to reshape this. but first need to swap around
            # axes due to how reshape works (it iterates over last 
            # dimension first). in particular, e.g.,:
            #       (32 x 200 x 3 x 1) -> (32 x 3 x 200 x 1)
            # swapped = np.swapaxes(X, 1, 2)
            swapped_weights = np.swapaxes(weights, 1, 2)
            init_weights = swapped_weights.reshape(self.n_filters, 
                            1, 1, n_gram*self.preprocessor.embedding_dims)

            cur_conv = Convolution2D(self.n_filters, 1, 
                                     n_gram*self.preprocessor.embedding_dims, 
                                     subsample=(1, self.preprocessor.embedding_dims),
                                     name="conv2d_"+str(n_gram), activation="relu",
                                     weights=[init_weights, biases])(x)

            # this output (n_filters x max_doc_len x 1)
            one_max = MaxPooling2D(pool_size=(1, self.preprocessor.max_sent_len-n_gram+1), 
                                   name="pooling_"+str(n_gram))(cur_conv)

            # flip around, to get (1 x max_doc_len x n_filters)
            permuted = Permute((2,1,3), name="permuted_"+str(n_gram)) (one_max)
            
            # drop extra dimension
            r = Reshape((self.preprocessor.max_doc_len, self.n_filters), 
                            name="conv_"+str(n_gram))(permuted)
            
            convolutions.append(r)


        sent_vectors = merge(convolutions, name="sentence_vectors", mode="concat")
        sent_vectors = Dropout(self.sent_dropout, name="dropout")(sent_vectors)

        # now we take a weighted sum of the sentence vectors to induce a document representation
        sent_sm_weights, sm_biases = self.sentence_model.get_layer("sentence_prediction").get_weights()

        print("end-to-end training is: %s" % self.end_to_end_train)
        sent_pred_model = Dense(3, activation="softmax", name="sentence_prediction", 
                                    weights=[sent_sm_weights, sm_biases], 
                                    trainable=self.end_to_end_train)

        # note that using the sent_preds directly works as expected...
        sent_preds = TimeDistributed(sent_pred_model, name="sentence_predictions")(sent_vectors)
  
        sw_layer = Lambda(lambda x: K.max(x[:,0:2], axis=1), output_shape=(1,)) 
        sent_weights = TimeDistributed(sw_layer, name="sentence_weights")(sent_preds)
 
        def scale_merge(inputs):
            sent_vectors, sent_weights = inputs[0], inputs[1]
            return K.batch_dot(sent_vectors, sent_weights)

        def scale_merge_output_shape(input_shape):
            # this is expected now to be (None x sentence_vec_length x doc_length)
            # or, e.g., (None, 96, 200)
            input_shape_ls = list(input_shape)[0]
            # should be (batch x sentence embedding), e.g., (None, 96)
            return (input_shape_ls[0], input_shape_ls[1])


        # sent vectors will be, e.g., (None, 200, 96)
        # -> reshuffle for dot product below in merge -> (None, 96, 200)
        sent_vectors = Permute((2, 1), name="permuted_sent_vectors")(sent_vectors)
        doc_vector = merge([sent_vectors, sent_weights], 
                                        name="doc_vector",
                                        mode=scale_merge,
                                        output_shape=scale_merge_output_shape)


        # trim extra dim
        doc_vector = Reshape((total_sentence_dims,), name="reshaped_doc")(doc_vector)
        doc_vector = Dropout(self.doc_dropout, name="doc_v_dropout")(doc_vector)

        output = Dense(1, activation="sigmoid", name="doc_prediction")(doc_vector)
        
        # ... and compile
        self.doc_model = Model(input=tokens_input, output=output)
        self.doc_model.compile(metrics=["accuracy"], loss="binary_crossentropy", optimizer="adadelta")
        print("rationale CNN model: ")
        print(self.doc_model.summary())
        

    def build_sentence_model(self):
        ''' 
        Build the *sentence* level model, which operates over, erm, sentences. 
        The task is to predict which sentences are pos/neg rationales.
        '''
        tokens_input = Input(name='input', shape=(self.preprocessor.max_sent_len,), dtype='int32')
        x = Embedding(self.preprocessor.max_features, self.preprocessor.embedding_dims, 
                      name="embedding",
                      input_length=self.preprocessor.max_sent_len, 
                      weights=self.preprocessor.init_vectors)(tokens_input)
        
        x = Dropout(self.sent_dropout)(x) # @TODO; parameterize! 

        convolutions = []
        for n_gram in self.ngram_filters:
            cur_conv = Convolution1D(name="conv_" + str(n_gram), 
                                         nb_filter=self.n_filters,
                                         filter_length=n_gram,
                                         border_mode='valid',
                                         activation='relu',
                                         subsample_length=1,
                                         input_dim=self.preprocessor.embedding_dims,
                                         input_length=self.preprocessor.max_sent_len)(x)
            # pool
            one_max = MaxPooling1D(pool_length=self.preprocessor.max_sent_len - n_gram + 1)(cur_conv)
            flattened = Flatten()(one_max)
            convolutions.append(flattened)

        sentence_vector = merge(convolutions, name="sentence_vector", mode="concat") # hang on to this layer!
        output = Dense(3, activation="softmax", name="sentence_prediction", 
                                W_constraint=maxnorm(9))(sentence_vector)

        self.sentence_model = Model(input=tokens_input, output=output)
        print("model built")
        print(self.sentence_model.summary())
        self.sentence_model.compile(loss='categorical_crossentropy', 
                                    metrics=['accuracy'], optimizer="adagrad")

        return self.sentence_model 


    def train_sentence_model(self, train_documents, nb_epoch=5, downsample=True, sent_val_split=.2, 
                                sentence_model_weights_path="sentence_model_weights.hdf5"):
        # assumes sentence sequences have been generated!
        assert(train_documents[0].sentence_sequences is not None)

        X, y= [], []
        # flatten sentences/sentence labels
        for d in train_documents:
            X.extend(d.sentence_sequences)
            y.extend(d.sentences_y)

        X, y = np.asarray(X), np.asarray(y)
        if downsample:
            X, y = RationaleCNN.balanced_sample(X, y)

        
        checkpointer = ModelCheckpoint(filepath=sentence_model_weights_path, 
                                       verbose=1, 
                                       save_best_only=True)
   
        self.sentence_model.fit(X, y, nb_epoch=nb_epoch, 
                                    callbacks=[checkpointer],
                                    validation_split=sent_val_split)

        # reload best weights
        self.sentence_model.load_weights(sentence_model_weights_path)

        self.sentence_model_trained = True


    def generate_sentence_predictions(documents, p):
        assert self.sentence_model_trained

        self.sentence_weights = []
        for d in documents:
            #x_doc = d.get_padded_sequences(p)

            sent_probs = self.sentence_model.predict(d.sentence_sequences)
            weights = np.amax(sent_probs[:,0:2],axis=1)
            #weights = np.amax(sentence_predictions[:,0:2],axis=1)

            # recall that we have padded the documents so there may be 
            # many 0 vector sentences at the end; we want the predicted
            # probabilities for these to be 0. 
            # sent_probs[d.num_sentences-1:]=np.zeros()
            d_probs = np.zeros(p.max_doc_len)
            d_probs[:d.num_sentences] = weights
            self.sentence_weights.append(d_probs)

        return d_probs



class Document:
    def __init__(self, doc_id, sentences, doc_label=None, sentences_labels=None):
        self.doc_id = doc_id
        self.doc_y = doc_label

        self.sentences = sentences
        self.sentence_sequences = None
        # length, pre-padding!
        self.num_sentences = len(sentences)

        self.sentences_y = sentences_labels
        self.sentence_weights = None 

        self.sentence_idx = 0
        self.n = len(self.sentences)


    def __len__(self):
        return self.n 

    def generate_sequences(self, p):
        ''' 
        p is a preprocessor that has been instantiated
        elsewhere! this will be used to map sentences to 
        integer sequences here.
        '''

        # here we build twice, which is supp
        self.sentence_sequences = p.build_sequences(self.sentences)

    def get_padded_sequences(self, p):
        # return p.build_sequences(self.sentences, pad_documents=True)              
        n_sentences = self.sentence_sequences.shape[0]
        X = self.sentence_sequences

        if n_sentences > p.max_doc_len:
            X = X[:p.max_doc_len]
        elif n_sentences < p.max_doc_len:
            # pad
            dummy_rows = np.zeros((p.max_doc_len-n_sentences, p.max_sent_len), dtype='int32')
            X = np.vstack((X, dummy_rows))

        return np.array(X)

class Preprocessor:
    def __init__(self, max_features, max_sent_len, embedding_dims=200, wvs=None, max_doc_len=500):
        '''
        max_features: the upper bound to be placed on the vocabulary size.
        max_sent_len: the maximum length (in terms of tokens) of the instances/texts.
        embedding_dims: size of the token embeddings; over-ridden if pre-trained
                          vectors is provided (if wvs is not None).
        '''

        self.max_features = max_features  
        self.tokenizer = Tokenizer(nb_words=self.max_features)
        self.max_sent_len = max_sent_len  # the max sentence length! @TODO rename; this is confusing. 
        self.max_doc_len = max_doc_len # w.r.t. number of sentences!

        self.use_pretrained_embeddings = False 
        self.init_vectors = None 
        if wvs is None:
            self.embedding_dims = embedding_dims
        else:
            # note that these are only for initialization;
            # they will be tuned!
            self.use_pretrained_embeddings = True
            self.embedding_dims = wvs.vector_size
            self.word_embeddings = wvs


    def preprocess(self, all_docs):
        ''' 
        This fits tokenizer and builds up input vectors (X) from the list 
        of texts in all_texts. Needs to be called before train!
        '''
        self.raw_texts = all_docs
        self.fit_tokenizer()
        if self.use_pretrained_embeddings:
            self.init_word_vectors()


    def fit_tokenizer(self):
        ''' Fits tokenizer to all raw texts; remembers indices->words mappings. '''
        self.tokenizer.fit_on_texts(self.raw_texts)
        self.word_indices_to_words = {}
        for token, idx in self.tokenizer.word_index.items():
            self.word_indices_to_words[idx] = token

    def build_sequences(self, texts, pad_documents=False):
        X = list(self.tokenizer.texts_to_sequences_generator(texts))
        # need to pad the number of sentences, too.
        X = np.array(pad_sequences(X, maxlen=self.max_sent_len))

        return X

    def init_word_vectors(self):
        ''' 
        Initialize word vectors.
        '''
        self.init_vectors = []
        unknown_words_to_vecs = {}
        for t, token_idx in self.tokenizer.word_index.items():
            if token_idx <= self.max_features:
                try:
                    self.init_vectors.append(self.word_embeddings[t])
                except:
                    if t not in unknown_words_to_vecs:
                        # randomly initialize
                        unknown_words_to_vecs[t] = np.random.random(
                                                self.embedding_dims)*-2 + 1

                    self.init_vectors.append(unknown_words_to_vecs[t])

        # note that we make this a singleton list because that's
        # what Keras wants. 
        self.init_vectors = [np.vstack(self.init_vectors)]
