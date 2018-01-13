import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing import text, sequence
from keras.layers import Flatten, Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.regularizers import Regularizer
from keras.callbacks import TensorBoard, ModelCheckpoint

from keras import backend

train  = pd.read_table('../input/train.tsv', sep='\t')# ,nrows=10000)
test = pd.read_table('../input/test.tsv', sep='\t')#, nrows=10000)

train_n = train.shape[0]
test_n = test.shape[0]

train.price = np.log1p(train.price)
    
nan_feats = ['category_name', 'brand_name', 'item_description']

for col in nan_feats:
    train[col].fillna('missing', inplace=True)
    test[col].fillna('missing', inplace=True)
 
le = LabelEncoder()
le.fit(np.hstack([train.category_name, test.category_name]))
train['category_name_labeled'] = le.transform(train.category_name)
test['category_name_labeled'] = le.transform(test.category_name)

le.fit(np.hstack([train.brand_name, test.brand_name]))
train.brand_name = le.transform(train.brand_name)
test.brand_name = le.transform(test.brand_name)

le.fit(np.hstack([train.shipping, test.shipping]))
train.shipping = le.transform(train.shipping)
test.shipping = le.transform(test.shipping)


### tokenization & pad 
        
tok = text.Tokenizer()

raw_text = np.hstack([train.name, train.item_description, 
                      test.name, test.item_description])
        
tok.fit_on_texts(raw_text)

vocab_size = len(tok.word_index) + 1 

train.name = tok.texts_to_sequences(train.name)
train.item_description = tok.texts_to_sequences(train.item_description)

test.name = tok.texts_to_sequences(test.name)
test.item_description = tok.texts_to_sequences(test.item_description)

### max_name, max_item_description chosen from histograms
#np.histogram([len(name) for name in train.name])
#np.histogram([len(desc) for desc in train.item_description])
#max_category_name = 10 #20
max_name = 10 #20
max_item_description = 75 #60


X_train, X_valid, y_train, y_valid = train_test_split(train, train['price'], 
                                                      test_size=0.01, random_state=777)

def get_keras_dict(df):
    X = {'category_name_labeled' : np.array(df.category_name_labeled),
         'brand_name' : np.array(df.brand_name),
         'shipping' : np.array(df.shipping),
         'item_condition_id' : np.array(df.item_condition_id),            
         'name' : sequence.pad_sequences(df.name, maxlen=max_name),
         'item_description' : sequence.pad_sequences(df.item_description, maxlen=max_item_description)   
     }
    return X

X_train = get_keras_dict(X_train)
X_valid = get_keras_dict(X_valid)
X_test = get_keras_dict(test)



def rmsle(y, y_pred):
    assert len(y) == len(y_pred)

    mse = sum((y_pred-y)**2) / len(y)
    
    return np.sqrt(mse)

def eval_model(model):
    val_preds = model.predict(X_valid)
    
    y_true = np.array(y_valid.values)
    y_pred = val_preds[:, 0]
    v_rmsle = rmsle(y_true, y_pred)
    print(" RMSLE error on dev test: "+str(v_rmsle))
    return v_rmsle

def get_model():  
    max_category_name_labeled = np.max([train.category_name_labeled.max(), test.category_name_labeled.max()])+1
    max_brand_name = np.max([train.brand_name.max(), test.brand_name.max()])+1
    
    category_name_labeled = Input(shape=[1], name='category_name_labeled')
    brand_name = Input(shape=[1], name='brand_name')
    shipping = Input(shape=[1], name='shipping')
    item_condition_id = Input(shape=[1], name='item_condition_id')
    
    name = Input(shape=[max_name], name='name') 
    item_description = Input(shape=[max_item_description], name='item_description')
    
    ##embeddings
    emb_category_name_labeled = Embedding(max_category_name_labeled, 10) (category_name_labeled)
    
    emb_brand_name = Embedding(max_brand_name, 10) (brand_name)
    emb_shipping = Embedding(2, 2) (shipping)
    max_item_condition_id = np.max([train.item_condition_id.max(), test.item_condition_id.max()])+1
    emb_item_condition_id = Embedding(max_item_condition_id, 5) (item_condition_id)    
    
    
    emb_name = Embedding(vocab_size+1, max_name) (name)
    emb_item_description = Embedding(vocab_size+1, max_item_description) (item_description)
    
    convs1 = []
    convs2 = []
    
    for filter_length in [1,2]:
        cnn_layer1 = Conv1D(filters=50, kernel_size=filter_length, padding='same', activation='relu', strides=1) (emb_name)
        cnn_layer2 = Conv1D(filters=50, kernel_size=filter_length, padding='same', activation='relu', strides=1) (emb_item_description)
        
        maxpool1 = GlobalMaxPooling1D() (cnn_layer1)
        maxpool2 = GlobalMaxPooling1D() (cnn_layer2)
        
        convs1.append(maxpool1)
        convs2.append(maxpool2)

    convs1 = concatenate(convs1)
    convs2 = concatenate(convs2)
    
    main_l = concatenate([
            
            Flatten() (emb_category_name_labeled),
            Flatten() (emb_brand_name),
            Flatten() (emb_shipping),
            Flatten() (emb_item_condition_id),
            
            convs1, 
            convs2, 
            
    ])
    main_l = Dropout(0.25)(Dense(128, activation='relu') (main_l)) #.25 = .435
    main_l = Dropout(0.1)(Dense(64, activation='relu') (main_l)) #.1
    
    # , kernel_regularizer=keras.regularizers.l2(0.01)
    output = Dense(1, activation='linear') (main_l)

    model = Model([category_name_labeled, brand_name, shipping, item_condition_id,
                   name, item_description], output)
    model.compile(loss='mse', optimizer='adam')
    
    return model


## callbacks
cp_fpath = 'checkpoint.hdf5' #'../output/keras-epoch_{epoch:02d}-vloss_{val_loss:.4f}.hdf5'
checkpoint_cb = ModelCheckpoint(cp_fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensor_cb = TensorBoard(log_dir='../graph-logs', histogram_freq=0, write_graph=True,)
#tb_cb.set_model(model)

model = get_model()

#backend.set_value(model.optimizer.lr, 0.005)
#model.load_weights('checkpoint.hdf5')

model.fit(X_train, y_train, 
          epochs=3, 
          batch_size=512,  
          #callbacks=[checkpoint_cb],
          validation_data=(X_valid, y_valid),
        )
v_rmsle = eval_model(model)

preds = model.predict(X_test, batch_size=20)
preds = np.expm1(preds)

submission = test[["test_id"]].copy() 
submission["price"] = preds
submission.to_csv("keras-cnn"+"_{:.6}.csv".format(v_rmsle), index=False)
