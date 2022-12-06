import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn import metrics
import joblib
import altair as alt

st.write(""" 
# Prediksi Kualitas dari Wine (Good/Bad)
Oleh Ahmad Dani Kurniawan (200411100205)
""")

st.write("===="*20)

tab0, tab1, tab2, tab3, tab4, tab5= st.tabs(["Description","Import Data", "Preprocessing", "Modelling", "Implementation", "Evaluation"])

with tab0:
    st.write("# Deskripsi Dataset")
    st.write("Kumpulan data ini terkait dengan varian merah anggur Vinho Verde Portugis. Kumpulan data menjelaskan jumlah berbagai bahan kimia yang ada dalam anggur dan pengaruhnya terhadap kualitasnya. Kumpulan data dapat dilihat sebagai tugas klasifikasi atau regresi. Kelas diurutkan dan tidak seimbang (Misalnya menentukan kualitas dari wine apakah memiliki kualitas yang baik atau tidak).")

    st.write("## Fitur yang dibutuhkan")
    st.write("1. Keasaman Tetap, dimana setiap anggur merah akan diujikan keasaman tetap apakah itu tingkat asam yang tinggi atau rendah")
    st.write("2. Keasaman Volatil, dimana setiap anggur merah akan diujikan keasaman volatil apakah itu tingkat asam yang tinggi atau rendah")
    st.write("3. Asam Sitrat, dimana setiap anggur merah akan diujikan Asam Sitrat apakah itu tingkat asam yang tinggi atau rendah")
    st.write("4. Kadar Gula, dimana setiap anggur merah akan diujikan Kadar Gula yang terkandung apakah itu tingkat Kadar Gula yang tinggi atau rendah")
    st.write("5. Kadar Klorida, dimana setiap anggur merah akan diujikan Kadar Klorida yang terkandung apakah itu tingkat kadar Klorida yang tinggi atau rendah")
    st.write("6. Kadar Sulfur dioksida bebas, dimana setiap anggur merah akan diujikan Kadar Sulfur dioksida bebas yang terkandung apakah itu tingkat Sulfur dioksida yang tinggi atau rendah")
    st.write("7. Total Sulfur dioksida, dimana setiap anggur merah akan diujikan Total Sulfur dioksida yang terkandung apakah itu Total Sulfur dioksida yang tinggi atau rendah")
    st.write("8. Tingkat Kepadatan, dimana setiap anggur merah akan diujikan Tingkat Kepadatan yang terkandung apakah itu Tingkat Kepadatan yang tinggi atau rendah")
    st.write("9. Total pH, dimana setiap anggur merah akan diujikan Total pH yang terkandung apakah itu Total pH yang tinggi atau rendah")
    st.write("10. Total Sulfat, dimana setiap anggur merah akan diujikan Total Sulfat yang terkandung apakah itu Total Sulfat yang tinggi atau rendah")
    st.write("11. Kadar Alkohol, dimana setiap anggur merah akan diujikan Kadar Alkohol yang terkandung apakah itu Kadar Alkohol yang tinggi atau rendah")

    st.write("## Output yang akan keluar")
    st.write("Terdapat 2 nilai output yakni Kualiatas Wine Baik dan Kualitas Wine Buruk")
    
with tab1:
    st.write("Import Data")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data)

with tab2:
    mm = st.checkbox('Normalisasi MixMax')

    if mm:
        data.head()
        wine = data.drop(columns=["Id"])

        sebelum_normalisasi = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
        sesudah_normalisasi = ['new_fixed acidity', 'new_volatile acidity', 'new_citric acid', 'new_residual sugar', 'new_chlorides', 'new_free sulfur dioxide', 'new_total sulfur dioxide', 'new_density', 'new_pH', 'new_sulphates', 'new_alcohol']
        normalize_feature = data[sebelum_normalisasi]

        st.write("## dataset sebelum normalisasi")

        scaler = MinMaxScaler()
        scaler.fit(normalize_feature)
        fitur_ternormalisasi=scaler.transform(normalize_feature)

        # save normalisasi

        joblib.dump(scaler, 'normal')

        fitur_ternormalisasi_df = pd.DataFrame(fitur_ternormalisasi, columns = sesudah_normalisasi)

        st.write("Data yang telah dinormalisasi")
        st.dataframe(fitur_ternormalisasi)

        data_sudah_normal= wine.drop(columns=sebelum_normalisasi)

        data_sudah_normal= data_sudah_normal.join(fitur_ternormalisasi_df)

        st.dataframe(data_sudah_normal)

with tab3:
    st.write("# modeling")

    Y = data_sudah_normal['quality']

    X = data_sudah_normal.iloc[:,1:12]

    X_train, X_test, y_train, y_test=train_test_split(X, Y, test_size=0.33, random_state=42)

    ### Dictionary to store model and accuracy
    
    model_accuracy = OrderedDict()

    ### Dictionary to store model and precision

    model_precision = OrderedDict()

    ### Dictionary to store model and recall

    model_recall = OrderedDict()

    # Naive bayes
    naive_bayes_classifier = GaussianNB()
    naive_bayes_classifier.fit(X_train, y_train)
    Y_pred_nb = naive_bayes_classifier.predict(X_test)

    # Decision Tree
    clf_dt = DecisionTreeClassifier(criterion="gini")
    clf_dt = clf_dt.fit(X_train, y_train)
    Y_pred_dt = clf_dt.predict(X_test)

    #K-Nearest Neighboor
    k_range = range(1,26)
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        Y_pred_knn = knn.predict(X_test)
    
    # Bagging Decision tree
    clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=10, random_state=0).fit(X_train, y_train)
    rsc = clf.predict(X_test)
    c = ['Naive Bayes']
    tree = pd.DataFrame(rsc,columns = c)

    # Random Forest
    rf = RandomForestClassifier(criterion='entropy')
    rf = rf.fit(X_train, y_train)
    Y_pred_rf = rf.predict(X_test)

    # save model dengan akurasi tertinggi
    joblib.dump(rf, 'Random_Forest')

    naive_bayes_accuracy = round(100 * accuracy_score(y_test, Y_pred_nb), 2)
    model_accuracy['Gaussian Naive Bayes'] = naive_bayes_accuracy
    decision_tree_accuracy = round(100 * metrics.accuracy_score(y_test, Y_pred_dt))
    knn_accuracy = round(100 * accuracy_score(y_test, Y_pred_knn), 2)
    bagging_Dc = round(100 * accuracy_score(y_test, tree), 2)
    random_forest_accuracy = round(100 * accuracy_score(y_test, Y_pred_rf))

    st.write("Pilih Metode : ")
    naive_bayes_cb = st.checkbox("Naive Bayes")
    decision_tree_cb = st.checkbox("Decision Tree")
    knn_cb = st.checkbox("K-Nearest Neighboor")
    bagging_tree_cb = st.checkbox("Bagging Decision Tree")
    random_forest_cb = st.checkbox("Random Forest")

    if naive_bayes_cb:
        st.write('Akurasi Metode Naive Bayes {} %.'.format(naive_bayes_accuracy))
    if decision_tree_cb:
        st.write('Akurasi Metode Decision Tree {} %.'.format(decision_tree_accuracy))
    if knn_cb:
        st.write('Akurasi Metode KNN {} %.'.format(knn_accuracy))
    if bagging_tree_cb:
        st.write('Akurasi Metode Bagging Decision Tree {} %.'.format(bagging_Dc))
    if random_forest_cb:
        st.write('Akurasi Metode Random Forest {} %.'.format(random_forest_accuracy))

with tab4:
    st.write("# Implementation")
    nama_wine = st.text_input("Masukkan Nama Wine")
    fixed_acidity = st.number_input("Masukkan Tingkat Keasaman Tetap ", min_value=4.60, max_value=15.60)
    volatile_acidity = st.number_input("Masukkan Tingkat Keasaman Volatil ", min_value=0.12, max_value=1.58)
    citric_acid = st.number_input("Masukkan Tingkat Asam Sitrat ", min_value=0.00, max_value=1.00)
    residual_sugar = st.number_input("Masukkan Kadar Sisa Gula ", min_value=0.90, max_value=15.50)
    chlorides = st.number_input("Masukkan Kadar Klorida ", min_value=0.012, max_value=0.611)
    free_sulfur_dioxide = st.number_input("Masukkan Kadar Sulfur Dioksida Bebas ", min_value=1.00, max_value=68.00)
    total_sulfur_dioxide = st.number_input("Masukkan Total Sulfur Dioksida ", min_value=6.00, max_value=289.00)
    density = st.number_input("Masukkan Tingkat Kepadatan ", min_value=0.9901, max_value=1.0037)
    p_H = st.number_input("Masukkan Total PH ", min_value=2.740, max_value=4.010)
    sulphates = st.number_input("Masukkan Total Sulfat ", min_value=0.33, max_value=2.00)
    alcohol = st.number_input("Masukkan Total Kadar Alkohol ", min_value=8.40, max_value=14.90)
    ##bagan = pd.DataFrame({'Akurasi ' : [naive_bayes_accuracy, decision_tree_accuracy, knn_accuracy, bagging_Dc, random_forest_accuracy], 'Metode' : ["Naive Bayes", "Decision Tree", "K-Nearest Neighboor", "Bagging Decision Tree", "Random Forest"]})

    st.write("Cek apakah wine masuk kategori good atau bad")
    cek_random_forest = st.button('Cek Wine')
    inputan = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, p_H, sulphates, alcohol]]

    scaler_jl = joblib.load('normal')
    scaler_jl.fit(inputan)
    inputan_normal = scaler.transform(inputan)

    FIRST_IDX = 0
    random_forest_1 = joblib.load("Random_Forest")
    if cek_random_forest:
        hasil_test = random_forest_1.predict(inputan_normal)[FIRST_IDX]
        if hasil_test >= 4:
            st.write("Nama Wine ", nama_wine , "Memiliki Kualitas Baik Berdasarkan Model random forest")
        else:
            st.write("Nama Wine ", nama_wine , "Memiliki Kualitas Buruk Berdasarkan Model random forest")

with tab5:
    st.write("# EVALUATION")
    bagan = pd.DataFrame({'Akurasi ' : [naive_bayes_accuracy, decision_tree_accuracy, knn_accuracy, bagging_Dc, random_forest_accuracy], 'Metode' : ["Naive Bayes", "Decision Tree", "K-Nearest Neighboor", "Bagging Decision Tree", "Random Forest"]})

    bar_chart = alt.Chart(bagan).mark_bar().encode(
        y = 'Akurasi ',
        x = 'Metode',
    )

    st.altair_chart(bar_chart, use_container_width=True)