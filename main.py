import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import tensorflow as tf
import ktrain
from ktrain import text
import seaborn as sns
import re
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import PhotoImage
import threading

data=None ### global değişken

############## CSV yükleme ################
def load_csv():
    global data
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            data = pd.read_csv(file_path,encoding='latin-1')
            label.config(text="Dosya başarıyla yüklendi: ")

        except Exception as e:
            label.config(text="Hata: Dosya yüklenemedi. Hata Detayı: " + str(e))



############## Button Configs ###############

def disable_buttons():
    load_button.config(state="disabled")
    xgboost_button.config(state="disabled")
    bert_button.config(state="disabled")

def enable_buttons():
    load_button.config(state="normal")
    xgboost_button.config(state="normal")
    bert_button.config(state="normal")


############## XGBoost #####################
def xgboost():
    disable_buttons()
    global data
    if data is None:
        label_xgboost.config(text="Veri yüklenemedi.")
        messagebox.showwarning("Uyarı", "Geçerli bir CSV dosyası seçin.")
        enable_buttons()
        return

    # Veri yükleme
    # data = pd.read_csv("C:/Users/bckal/PycharmProjects/pythonProject/spam.csv", encoding='latin-1')

    # Veriyi eğitim ve test setlerine bölmek
    X = data['v2']
    y = data['v1']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 'ham' sınıfını 0, 'spam' sınıfını 1 olarak dönüştürme
    y_train = y_train.map({'ham': 0, 'spam': 1})
    y_test = y_test.map({'ham': 0, 'spam': 1})

    # TF-IDF vektörize ediciyi tanımlama
    vectorizer = TfidfVectorizer()

    # 'v2' sütununu TF-IDF vektörlerine dönüştürme
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # XGBoost modelini tanımlama ve eğitme
    clf = XGBClassifier(max_depth=3, n_estimators=1000, learning_rate=0.01)
    clf.fit(X_train_tfidf, y_train)

    # Modelin performansını test seti üzerinde değerlendirme
    accuracy = clf.score(X_test_tfidf, y_test)
    #print("Doğruluk puanı:", accuracy)
    label_xgboost.config(text="Modelin doğruluk puanı: {}".format(accuracy))

    # TF-IDF vektörize ediciyi tanımlama
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Kullanıcıdan metin girişini kontrol edelim
    text = text_box.get("1.0", "end-1c")  # Metin kutusundan metni alıyoruz
    if not text:
        text_box.delete("1.0", "end")
        messagebox.showwarning("Uyarı", "Lütfen metin giriniz...")
        enable_buttons()
    else:
        # Kullanıcının girdisini TF-IDF vektörüne dönüştürme
        text_tfidf = vectorizer.transform([text])

        # Modeli kullanarak tahmin yapma
        prediction = clf.predict(text_tfidf)[0]

        # Tahmin sonucunu ekrana basalım
        if prediction == 0:
            messagebox.showwarning("Uyarı", "Girdiğiniz metin 'Ham' bir mesaj olarak tahmin edildi.")
            enable_buttons()
        else:
            messagebox.showwarning("Uyarı", "Girdiğiniz metin 'Spam' bir mesaj olarak tahmin edildi.")
            enable_buttons()








############### BERT ########################

def bert():
    disable_buttons()
    global data
    if data is None:
        label_bert.config(text="Veri yüklenemedi.")
        messagebox.showwarning("Uyarı", "Geçerli bir CSV dosyası seçin.")
        enable_buttons()
        return
    df=data
    #df = pd.read_csv("C:/Users/bckal/PycharmProjects/pythonProject/spam.csv", encoding='ISO-8859-1')
    df.info()
    df.drop(df.iloc[:, 2:], inplace=True, axis=1)
    df.isnull().sum()
    df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
    #print('The total number of spam and not spam message in our dataset is\n', df['label'].value_counts())

    # sns.countplot(df['label'])
    split = int(len(df) * 0.90)
    train_data = df.iloc[:split, :]
    test_data = df.iloc[split:, :]

    train_data

    (X_train, y_train), (X_test, y_test), preprocess = text.texts_from_df(train_df=train_data,
                                                                          text_column='message', label_columns='label',
                                                                          val_df=test_data, maxlen=300,
                                                                          preprocess_mode='bert')

    X_train[0].shape

    X_train[0].ndim

    X_train

    model = text.text_classifier(name='bert', train_data=(X_train, y_train), preproc=preprocess, metrics=['accuracy'])

    learner = ktrain.get_learner(model=model, train_data=(X_train, y_train), val_data=(X_test, y_test), batch_size=3)

    ############ devre dışı ############

    # learner.lr_find()
    # learner.lr_plot()
    # learner.fit_onecycle(lr=1e-5, epochs=1)
    # predictor=ktrain.get_predictor(learner.model,preprocess)

    predictor = ktrain.load_predictor('Bert_model')

    # Kullanıcıdan metin girişini kontrol edelim
    new_message = text_box.get("1.0", "end-1c")  # Metin kutusundan metni alıyoruz

    if new_message.strip() == "":

        text_box.delete("1.0", "end")
        text_box.insert("1.0", "Lütfen metin giriniz...")
        messagebox.showwarning("Uyarı", "Lütfen metin giriniz...")
        enable_buttons()
        return

    elif new_message and bert_checkbox_var.get():

        predictions = predictor.predict(new_message)
        # print(predictions)
        # Tahmin sonucunu ekrana basalım
        messagebox.showwarning("Uyarı", "Girdiğiniz metin " +predictions+ " bir mesaj olarak tahmin edildi.")
        messagebox.showwarning("Uyarı", "Doğruluk oranı hesaplanıyor. Bu işlem zaman alabilir.")

        # Mevcut modeli test veri seti üzerinde kullanarak tahminler yapın
        y_pred = predictor.predict(
            test_data['message'].tolist())  # Varsayılan olarak test veri setindeki mesaj sütununu kullanıyoruz

        # Gerçek etiketler
        y_true = test_data['label']

        # Doğruluk değerini hesaplayın
        accuracy = accuracy_score(y_true, y_pred)

        # Sonucu yazdırın
        messagebox.showwarning("Uyarı", "Doğruluk değeri: {}".format(accuracy))
        label_bert.config(text="Modelin doğruluk değeri: \n {}".format(accuracy))
        enable_buttons()
    else:
        predictions = predictor.predict(new_message)
        # print(predictions)
        # Tahmin sonucunu ekrana basalım
        messagebox.showwarning("Uyarı", "Girdiğiniz metin " + predictions + " bir mesaj olarak tahmin edildi.")
        enable_buttons()



    ######## Model önceden kaydedildi ############

    # predictor.save('Bert_model')

#################### Threading ##################

def bert_thread():
    thread = threading.Thread(target=bert)
    thread.start()

def xgboost_thread():
    thread = threading.Thread(target=xgboost)
    thread.start()

#################### GUI ########################

# Tkinter uygulamasını oluşturma
root = tk.Tk()
root.title("Spam Mesaj Kontrolü")

# İkonu belirle
root.iconbitmap("Mail.ico")

# Pencere boyutunu ayarlama
root.geometry("500x400")
root.resizable(False, False)

# Etiket
label = tk.Label(root, text="CSV dosyasını yüklemek için 'Yükle' butonuna tıklayın.", wraplength=400)
label.pack(pady=10)

# upload icon'u yükle

icon = PhotoImage(file="Upload.png")
# Resmi ekleyen bir Label oluştur
icon_label = tk.Label(root, image=icon)
icon_label.place(x=223,y=58)


# Yükleme düğmesi
load_button = tk.Button(root, text="Yükle", command=load_csv)
load_button.place(x=230,y=35)



# XGBoost butonu
xgboost_button = tk.Button(root, text="XGBoost", command=xgboost_thread)
xgboost_button.place(x=50, y=250)

# XGBoost etiketi
label_xgboost = tk.Label(root, text="")
label_xgboost.place(x=50, y=290)

# BERT butonu
bert_button = tk.Button(root, text="BERT", command=bert_thread)
bert_button.place(x=400, y=250)

# BERT etiketi
label_bert = tk.Label(root, text="")
label_bert.place(x=350, y=290)

# Onay kutusu (tick box)
bert_checkbox_var = tk.BooleanVar()
bert_checkbox = tk.Checkbutton(root, text="Doğruluk oranını hesapla", variable=bert_checkbox_var)
bert_checkbox.place(x=330, y=320)

# Metin kutusu
text_box = tk.Text(root, height=8, width=50)
text_box.place(x=50, y=110)

# Uygulamayı başlatma
root.mainloop()


