import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras import layers

df = pd.read_csv('data/apkfull.csv')
print(df.shape)
print(df.head())

rpl = df['RequestedPermissionList'].values
al = df['ActivityList'].values
sl = df['ServiceList'].values
cpl = df['ContentProviderList'].values
brl = df['BroadcastReceiverList'].values
hcl = df['HardwareComponentsList'].values
ifl = df['IntentFilterList'].values
upl = df['UsedPermissionsList'].values
ral = df['RestrictedApiList'].values
udl = df['URLDomainList'].values

y = df['label'].values

#loop train and test to find average performance
for loop in range(30):
    print("Current Loop is: ")
    print(loop)
    print("-----------------")

    from random import randint
    testNum = 0.33
    randomNum = randint(0, 1000000)
    #randomNum = 736744
    print("Current Random Seed: ")
    print(randomNum)

    #data split area

    rpl_train, rpl_test, y_train, y_test = train_test_split(
        rpl, y, test_size=testNum, random_state=randomNum)

    al_train, al_test, y_train, y_test = train_test_split(
        al, y, test_size=testNum, random_state=randomNum)

    sl_train, sl_test, y_train, y_test = train_test_split(
        sl, y, test_size=testNum, random_state=randomNum)

    cpl_train, cpl_test, y_train, y_test = train_test_split(
        cpl, y, test_size=testNum, random_state=randomNum)

    brl_train, brl_test, y_train, y_test = train_test_split(
        brl, y, test_size=testNum, random_state=randomNum)

    hcl_train, hcl_test, y_train, y_test = train_test_split(
        hcl, y, test_size=testNum, random_state=randomNum)

    ifl_train, ifl_test, y_train, y_test = train_test_split(
        ifl, y, test_size=testNum, random_state=randomNum)

    upl_train, upl_test, y_train, y_test = train_test_split(
        upl, y, test_size=testNum, random_state=randomNum)

    ral_train, ral_test, y_train, y_test = train_test_split(
        ral, y, test_size=testNum, random_state=randomNum)

    udl_train, udl_test, y_train, y_test = train_test_split(
        udl, y, test_size=testNum, random_state=randomNum)

    #vectorize the data area

    vectorizer = CountVectorizer(lowercase=False)
    vectorizer.token_pattern=u'(?u)"(.*?)"'     #for the apk unicode format of feature list
    print(vectorizer)

    #Fitting and NN setup area

    vectorizer.fit(rpl_train)
    # print(vectorizer.vocabulary_)
    x_rpl_train = pd.DataFrame(vectorizer.transform(rpl_train).toarray()) #matrix needs to be converted to array
    x_rpl_test = pd.DataFrame(vectorizer.transform(rpl_test).toarray())

    vectorizer.fit(al_train)
    # print(vectorizer.vocabulary_)
    x_al_train = pd.DataFrame(vectorizer.transform(al_train).toarray())
    x_al_test = pd.DataFrame(vectorizer.transform(al_test).toarray())

    vectorizer.fit(sl_train)
    #print(vectorizer.vocabulary_)
    x_sl_train = pd.DataFrame(vectorizer.transform(sl_train).toarray())
    x_sl_test = pd.DataFrame(vectorizer.transform(sl_test).toarray())

    vectorizer.fit(cpl_train)
    # print(vectorizer.vocabulary_)
    x_cpl_train = pd.DataFrame(vectorizer.transform(cpl_train).toarray())
    x_cpl_test = pd.DataFrame(vectorizer.transform(cpl_test).toarray())

    vectorizer.fit(brl_train)
    #print(vectorizer.vocabulary_)
    x_brl_train = pd.DataFrame(vectorizer.transform(brl_train).toarray())
    x_brl_test = pd.DataFrame(vectorizer.transform(brl_test).toarray())

    vectorizer.fit(hcl_train)
    # print(vectorizer.vocabulary_)
    x_hcl_train = pd.DataFrame(vectorizer.transform(hcl_train).toarray())
    x_hcl_test = pd.DataFrame(vectorizer.transform(hcl_test).toarray())

    vectorizer.fit(ifl_train)
    # print(vectorizer.vocabulary_)
    x_ifl_train = pd.DataFrame(vectorizer.transform(ifl_train).toarray())
    x_ifl_test = pd.DataFrame(vectorizer.transform(ifl_test).toarray())

    vectorizer.fit(upl_train)
    # print(vectorizer.vocabulary_)
    x_upl_train = pd.DataFrame(vectorizer.transform(upl_train).toarray())
    x_upl_test = pd.DataFrame(vectorizer.transform(upl_test).toarray())

    vectorizer.fit(ral_train)
    # print(vectorizer.vocabulary_)
    x_ral_train = pd.DataFrame(vectorizer.transform(ral_train).toarray())
    x_ral_test = pd.DataFrame(vectorizer.transform(ral_test).toarray())

    vectorizer.fit(udl_train)
    # print(vectorizer.vocabulary_)
    x_udl_train = pd.DataFrame(vectorizer.transform(udl_train).toarray())
    x_udl_test = pd.DataFrame(vectorizer.transform(udl_test).toarray())

    print("\n\n*****Data Seperation*****\n\n")

    #combining data for NN
    X_train = pd.concat([x_rpl_train, x_al_train, x_sl_train, x_cpl_train,
                         x_brl_train, x_hcl_train, x_ifl_train, x_upl_train,
                         x_ral_train, x_udl_train], axis=1)
    X_test  = pd.concat([x_rpl_test, x_al_test, x_sl_test, x_cpl_test,
                         x_brl_test, x_hcl_test, x_ifl_test, x_upl_test,
                         x_ral_test, x_udl_test], axis=1)

    #Keras Area
    input_dim = X_train.shape[1]  # Number of features

    print(input_dim)
    model = Sequential()
    model.add(layers.Dense(1000, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                 optimizer='SGD',
                 metrics=['accuracy'])
    model.summary()

    class_weights = {0: 0.5, 1: 6}
    history = model.fit(X_train, y_train,
                    epochs=100,
                    verbose=2,
                    validation_data=(X_test, y_test),
                    batch_size=32,
                    class_weight=class_weights)

    # prints NN architecture
    #from keras.utils import plot_model
    #plot_model(model, to_file='model.png', show_shapes=True)

    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    # more metrics
    from sklearn.metrics import classification_report

    target_names = ['Benign', 'Malicous']
    y_pred = model.predict_classes(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))

    #print metrics to file
    import time
    with open("log.txt", "a") as myfile:

        myfile.write("\n" + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        myfile.write(" - Random Seed: " + str(randomNum))
        myfile.write("\n" + classification_report(y_test, y_pred, target_names=target_names) + "\n")

    #visualization area
    import matplotlib.pyplot as plt
    import tkinter
    plt.style.use('ggplot')

    def plot_history(history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

    plot_history(history)

print("\n\n***end***")
with open("NN_metric_log.txt", "a") as myfile:
    myfile.write("\n***end***\n")
