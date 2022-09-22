# Laporan Proyek Machine Learning - Muhamad Gatot Supiadin

## Domain Proyek

Domain yang dipilih dalam proyek machine learning ini adalah Investment & financial, dengan judul **Predictive Analytics of GOOGL stock**

### Latar Belakang 
Pada dasarnya dunia investasi tidak jauh dari kata saham, coin, crypto, emas dan banyak lainya yang khususnya trend dikalangin gen-Z sekarang, orang yang sudah kenal dunia investasi sejak lama biasanya hidupnya sudah hampir bisa tercukupi untuk dimasa yang akan datang.

Orang yang berinvestasi pada sebuah company itu merupakan salah satu orang yang sudah peduli tentang masa depanya entah itu investasi dalam banyak hal properti ataupun saham misalnya, beberapa tahun belakangan ini kususnya saham teknologi company, hampir bisa menjamin hari tua seseorang jika dia berinvestasi pada company yang tepat, contohnya **Tesla** kemarin yang melonjak naik ratusan persen dikarenakan teknologi yang ia munculkan terlebih lagi **Twiitter** sudah diakuisisi menjadi pemilik penuh dari seorang **Elon Musk** sekaligus pemilik Tesla. jauh melenceng dari itu **GOOGL** juga menjadi pesaing yang lumayan bersaing dalam dunia saham, dengan search engine andalanya ia bisa meraup banyak sekali keuntungan dan jika kita berinvestasi didalamnya kitapun akan mendapatkan dampaknya yang bisa naik dalam beberapa tauhn kedepan dalam banyak saham dunia yang dimiliki company.

Oleh karena itu proyek ini akan mempermudah para investor muda untuk berinvestasi lebih pintar pada company yang tidak akan merugikan dalam rentan waktu beberapa tahun kedepan menggunakan Machine Learning secara conitnu atau bisa kita sebut Time Series Forecasting / Regression.

Forecasting merupakan bagian dari Machine Learning dimana itu salah teknik yang dapat meramalkan keadaan, harga dimasa yang akan datang menggunakan data data historynya. hal ini masih termasuk kedalam Time Series forecasting, dengan mendeteksi pola dan kecenderungan data historinya yang kemudian diformulasikan kedalam model machine learnign yang nanti akan kita gunakan untuk meprediksi data yang akan datang beberapa tahun kedepan.

## Business Understanding

### Problem Statements
Berdasarkan latar belakang yang telah dijelaskan diawal, berikut beberapa permasalahan yang dapat diselesaikan dalam proyek ini :
- Apakah **GOOGL** stock merupakan wadah yang baik untuk investasi jangka panjang, lalu bagaimana cara menganalisis harga dari stock **GOOGL**? 
- Bagaimana memilih algortma dan melatih model terbaik untuk data tersebut?
- Bagaimana cara menganalisis dan memprediksi harga stock GOOGL menggunakan Forecasting dalam Time Series?

### Goals
Tujuan dibuatnya proyek ini sebagai berikut :
- **GOOGL** menjadi jawaban positif untuk investor muda dalam berinvestasi untuk hari tua, serta harga dapat dianalisis menggunakan model machine learning.
- Melakukan training model terhadap beberapa algoritma dan memilih yang terbaik.
- Tentu saja ini berita bagus untuk para investor, mereka dapat berinvestasi ke sebuah company yang tanpa adanya keraguan dalam berinvestasi dimasa yang akan datang.


### Solution statements
Solusi yang bisa dilakukan agar goals dapat terpenuhi sebagai berikut :
* Melakukan analisa dan eksplorasi lebih jauh pada dataset dan memvisualisasikanya agar mendapat gambaran yang kuat. berikut merupakan tahapan yang bisa mewakili solution statement
  - Menangani jika terjadinya missing value pada data.
  - Mencari korelasi pada dataset untuk mencari dimana variabel dependent dan variabel  independent. 
  - Jika terdapat outliner, menganganinya dengan metode IQR.
  - Melakukan Normalization pada dataset terutama pada fitur numerik.
  - Membuat model regresi guna meprediksi bilangan kontinu harga saham dimasa yang akan datang.

* Berikut merupakan list algortima yang dicoba dalam model:
  - Support Vector Machine (Support Vector Regression)
  - K-Nearest Neighbors (KNN)
  - Boosting Algorithm (Gradient Boosting Regression)

* Seletah semua dilalui kita bisa menambahkan Hyperparameter tuning agar model dapat berjalan dalam performa terbaiknya menggunakan teknik Grid Search.

## Data Understanding
Pada proyek ini saya mengambil dataset publik dari Kaggle yang berjudul _Goggle Stock Data_ (https://www.kaggle.com/datasets/varpit94/google-stock-data).

Dataset yang digunakan memiliki format .csv, mempunyai total 4431 data dengan 7 kolom diantaranya (Date, Open, High, Low, Close, Adj Close dan Volume), berikut merupakan penjelasan masing masing kolom:
- Date : Opening rekap data
- High : Highest price per day
- Low : Lowest price per day
- Open : Opening price per day
- Close : Closing price per day
- Adj Close : Closing price per day after counting stock split or stock reverse
- Volume : Volume Transaction price per day

### Eksploratory Data Analysis
sebelum beranjak ke Data Preparation, kita harus mengetahui data, seperti korelasi, outliner, dan analisis Univariate dan Multivariate anailisis
- Mengangani adanya Outliner 

![outliner_before](https://user-images.githubusercontent.com/73319544/191768099-69c950a6-6e10-43fb-b13e-33ee38293c0d.png)

Gambar 1.Outliner Dataset

Pada gambar 1, Jika data numerik divisualisasikan, hanya fitur *Volume* saja yang memiliki banyak outlier. Untuk menangani outlier dengan menggunakan IQR Method yaitu dengan menghapus data yang berada diluar IQR yaitu antara 25% dan 75%. setela selesai kita akan mendapatkan total sampel data yang sudah clean terhadap outliner sebanyak 3861 data dan 7 kolom.

- Unvariate Analysis

![unvariative](https://user-images.githubusercontent.com/73319544/191768236-ee017345-dd9f-41a5-abd5-33642f5f9321.png)

Gambar 2.Unvariate Analysis

Pada Gambar 2, Karena yang kita cari adalah Adj Close, maka kita akan fokus ke salah satu kolom dimana kolomnya adalah kolom Adj Close

- Multivariate Analysis

Pada kali ini kita akan menganalisis korealsi Adj Close terhadap fitur lain. dan dapat disimpulkan bahwa Adj Close memiliki korelasi positif yang kuat terhadap kolom *Open, High, Low * dan * Close *, sedangkan terhadap kolom Volume tidak memiliki korelasi yang kuat.

![multivariate](https://user-images.githubusercontent.com/73319544/191768276-d1524ffe-0223-4a05-8882-a07df27f43f9.png)

Gambar 3.Multivariate Analysis

Pada Gambar 3, Kita bisa lihat dan berfokus pada kolom Adj Close dengan kolom lainya, pada plot diatas kita bisa berfokus pada plot baris ke-5. terlihat cukup jelas bahwa kolom Adj Close memiliki korelasi positif terhadap kolom High, Open, Low, dan Close. sementara pada kolom Volume memiliki korelasi yang lemah. 

Selanjutnya kita bisa membuat heatmap korelasi pada data menggunakan library Seaborn

![heatmap_correlation](https://user-images.githubusercontent.com/73319544/191768317-577fdfc2-69e4-43f2-a853-9707fd386938.png)

Gambar 4. Korelasi Heatmap

Pada Gambar 4, untuk memperjelas korelasi kita menggunakan heatmap dan kita tahu bahwa kolom Adj Close memiliki korelasi positif yang kuat terhadap setiap fitur yang ditandakan dengan angka 1 yang berarti kuat dan bisa kita gunakan sebagai Dependent variable terkecuali pada fitur Volume yang mempunyai korelasi lemah.

## Data Preparation
Pada tahap ini saya melakukan beberapa tahapan dalam pemrosesan data:

### Melakukan Penanganan Missing Value
Tahapan awal adalah menghilangkan Missing value pada dataset yang memiliki 2 cara untuk dihapus atau akan diisi dengan nilai rata rata menggunakan library Simpleimputer, karena dataset yang saya gunakan tidak memilki missing value kita bisa lanjut ke tahap selanjutnya

### Splitting dataset
Pada tahap ini dataset yang tadi kita sudah diolah kemudian kita bagi menjadi data train dan data test, dengan ratio yang bisa kita atur sendiri, pada proyek ini saya memberikan ratio 80:20, dimana 80% merupakan data train dan 20% merupakan data test yang sudah dibagi menggunakan librari train_test_split dari Sklearn.

### Menghapus fitur yang tidak diperlukan 
setelah diolah ternyata kita hanya memerlukan kolom Open, High, Low, dan Adj Close. oleh karena itu kita bisa drop atau menghapus fitur selain kolom diatas seperti Volume, Date dan Close.

### Normalization Data
Pada tahap ini kita ingin agar model bekerja optimal dan maksimal, oleh karena itu kita akan mengtransformasi data dalam rentan angka 0 hingg1 1 dengan menggunakan MinMaxScaler.

## Modeling
Pada tahap ini kita menggunakan 3 buah algoritma diantaranya ada _Support Vector Regression, Gradient Boost dan KNN_.

### Support Vector Regression 

Algoritma ini hampir sama seperti SVM tetapi pada SVM biasa digunakan dalam klasifikasi. Pada SVM, algoritma tersebut berusaha mencari jalan terbesar yang bisa memisahkan sampel dari kelas berbeda, sedangkan SVR mencari jalan yang dapat menampung sebanyak mungkin sampel di jalan. Berikut merupakan Hyper Parameter yang digunakan dalam model: 
 - kernel : Hyperparameter ini biasa digunakan untuk menghitung kernel pada matriks sebelumnya.
 - C : Hyperparameter ini biasa digunakan untuk menukar klasifikasi yang benar dari contoh training terhadap maksimalisasi margin fungsi keputusan.
 - gamma : Hyperparameter ini biasa digunakan untk menetukan seberapa jauh pengaruh satu contoh pelatihan mencapai, dengan nilai rendah berarti jauh dan nilai tinggi berarti dekat.

#### kelebihan
- Lebih efektif pada data dimensi tinggi (data dengan jumlah fitur yang banyak)
- Memori lebih efisien karena menggunakan subset poin pelatihan

#### Kekurangan 
- Sulit dipakai pada data skala besar

### Gradient Boost

Gradient Boosting adalah algoritma machine learning yang menggunakan teknik ensembel learning dari decision tree untuk memprediksi nilai. Gradient Boosting sangat mampu menangani pattern yang kompleks dan data ketika linear model tidak dapat menangani. Untuk hyperparameter yang digunakan pada model ini ada 3 yaitu: 
- learning_rate : Hyperparameter training yang digunakan untuk menghitung nilai koreksi bobot padded pada waktu proses training. Umumnya nilai learning rate berkisar antara 0 hingga 1
- n_estimators : Jumlah tahapan boosting yang akan dilakukan pada model.
- criterion : Hyperparameter yang biasanya digunakan untuk menemukan fitur dan ambang batas optimal dalam membagi data

#### kelebihan
- Hasil pemodelan yang lebih akurat
- Model yang stabil dan lebih kuat (robust)
- Dapat digunakan untuk menangkap hubungan linear maupun non linear pada data

#### Kekurangan 
- Pengurangan kemampuan interpretasi model
- Waktu komputasi dan desain tinggi
- Tingkat kesulitan yang tinggi dalam pemilihan model

### K-Nearest Neighbors (KNN)

K-Nearest Neighbors merupakan algoritma machine learning yang bekerja dengan mengklasifikasikan data baru menggunakan kemiripan dengan tetangganya atau bisa dikatakan antara data baru dengan sejumlah data (k) pada data yang telah ada. Algoritma ini dapat digunakan untuk klasifikasi dan regresi. Untuk hyperparameter yang digunakan pada model ini hanya 1 yaitu :

- n_neighbors : Parameter yang menunjukanJumlah tetangga untuk yang diperlukan untuk menentukan letak data baru

#### kelebihan
- Dapat menerima data yang masih noisy
- Sangat efektif apabila jumlah datanya banyak
- Mudah diimplementasikan

#### Kekurangan 
- Sensitif pada outlier
- Rentan pada fitur yang kurang informatif

#### Pemakaian Algortima 
Untuk proyek kali ini kita akan menggunakan model K-Nearest Neighbors karena memiliki error (0.00001) yang paling sedikit daripada model yang lain. Namun tidak bisa dipungkiri model dari Gradient Boosting juga memiliki error (0.000011) yang hampir seperti KNN.

## Evaluation
Pada tahap evaluasi ini metrik yang digunakan adalah Mean Squared Error (MSE), dimana dia akan mengukur seberapa dekat garis pas dengan titik pada data. 

![mse_rumus](https://user-images.githubusercontent.com/73319544/191768488-1d350af9-cb15-4fe1-8cd5-7c8ded16aaf9.png)

Gambar 5. Rumus MSE

Pada Gambar 5, Kita bisa melihat perhitungan dari MSE dimana penjelasan dari variablenya sebagai berikut :
- n = Jumlah titik data
- Yi = nilai sesungguhnya 
- Yi_hat = nilai prediksi

berikut merupakan Visualisai dari hasil akurasi model :

![mse](https://user-images.githubusercontent.com/73319544/191768548-b62df2d8-c282-4dd2-a616-a99d8986b3e5.png)

Gambar 6. Hasil MSE dari model

Pada Gambar 6, Kita bisa lihat hasil dari MSE model, dimana semakin kecil MSE yang diperoleh oleh model maka akan semakin optimal algortima tersebut.

![mse_plot](https://user-images.githubusercontent.com/73319544/191768597-750f3732-6abc-47d9-9f44-1a5e1f03eea8.png)

Gambar 7. Plot Visualisasi MSE pada model

Pada Gambar 7, kita bisa melihat hasil plot dari model terhadap beberapa algortma yang digunakan, ada 2 algoritma yang sangat cocok untuk gunakan diantaranya ada KNN dan Gradient Boosting.

![mse_accuracy](https://user-images.githubusercontent.com/73319544/191768645-d4ce7412-9004-4bf8-9e77-c6e6752802e7.png)

Gambar 8. Hasil akurasi MSE model

Pada Gambar 8, Kita bisa melihat akurasi dari setiap algortima yagn digunakan dan kita bisa simpulkan bahwa KNN atau K-Nearest Neighbors merupakan algortma paling optimal untuk model. 

Pada proyek ini semua model berjalan dengan sangat baik dan maksimal dan hanya terdapat selisih sangat kecil diantara ketiganya akan tetapi kita akan memilih model yang paling tinggi akurasinya, dimana K-Nearest Neighbors (KNN) adalah algortima yang memiliki nilai tertinggi.

### Forecasting
pada tahap ini saya akan mencoba memprediksi menggunakan algortma yang kita pili diatas yaitu KNN dalam kurun waktu 30 hari kedepan 

![Prediksi](https://user-images.githubusercontent.com/73319544/191768702-9dc917e8-8ca1-4418-8552-129bba3524f8.png)

Gambar 9. Prediksi model kurun waktu 30 hari

Pada Gambar 9, Bisa kita lihat prediksi harga yang akan datang dalam kurun waktu 30 hari kedepan yang sudah diprediksi menggunakan KNN yang telah kita pilih sebagai algortima yang paling optimal.

## Referensi :
* Sidhu, R. (sep 30, 2019).KNN Classification Algorithm in Python.Medium, from https://medium.com/x8-the-ai-community/knn-classification-algorithm-in-python-65e413e1cea0
* Saputri, L. (2016). IMPLEMENTASI JARINGAN SARAF TIRUAN RADIAL BASIS FUNCTION (RBF) PADA PERAMALAN FOREIGN EXCHANGE (FOREX).
* Investors' risk attitudes in the pandemic and the stock market: new evidence based on internet searches (26 June 2020), from https://www.bis.org/publ/bisbull25.htm
* Spiliotis E. Decision Trees for Time-Series Forecasting (january 2022), from https://www.researchgate.net/publication/359865759_Decision_Trees_for_Time-Series_Forecasting
*  Gudekar A. Stock Prediction Application using Machine Learning(April 2022) from https://www.researchgate.net/publication/360300426_Stock_Prediction_Application_using_Machine_Learning
