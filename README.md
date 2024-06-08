# 71220853_MatthewAlexander_PerawatanPesawat_Forecasting

# UAS_AI

Nama : Matthew Alexander

NIM : 71220853

# Prediksi Perawatan Pesawat 

Notebook yang saya buat menggunakan model AI  untuk prediksi perawatan pesawat dengan pendekatan Forecasting menggunakan Linear Regression.

# Forecasting 

Dalam konteks perawatan pesawat, prediksi yang akurat mengenai kapan suatu perawatan harus dilakukan sangat penting untuk mencegah kegagalan dan memastikan keselamatan. Forecasting menggunakan data historis membantu dalam mengidentifikasi tren dan pola yang dapat digunakan untuk prediksi di masa depan.Dengan melakukan forecasting, perawatan dapat dijadwalkan secara proaktif, sehingga menghindari keterlambatan dan kerusakan yang lebih parah. Ini juga membantu dalam mengoptimalkan jadwal perawatan dan sumber daya.

# Linear Regression

Linear regression adalah model yang sederhana namun sering kali cukup efektif untuk banyak masalah prediksi. Model linear regression relatif cepat untuk dilatih dan dievaluasi, yang membuatnya cocok untuk iterasi cepat selama pengembangan model.




# Eksplorasi Data

Membaca dataset dari file CSV menggunakan pandas. Metode head() menampilkan lima baris pertama dari dataset untuk mendapatkan gambaran awal. Metode describe() memberikan statistik deskriptif dasar dari dataset, seperti mean, standar deviasi, dan lain-lain. Metode info() memberikan informasi umum tentang dataset, termasuk jumlah nilai null dan tipe data dari setiap kolom.

Fitur & Target

Memisahkan dataset menjadi dua bagian: fitur (X) dan target (y). 
Semua kolom kecuali kolom 'target' digunakan sebagai fitur, sementara kolom 'target' adalah nilai yang ingin diprediksi.

Data Test & Train

Membagi data menjadi set pelatihan (training set) dan set pengujian (testing set) dengan proporsi 80:20. Parameter random_state digunakan untuk memastikan bahwa pembagian data konsisten di setiap eksekusi.

Standard Scaler

Menggunakan StandardScaler dari scikit-learn, yang mengubah data sehingga memiliki mean 0 dan standar deviasi 1. Ini penting untuk memastikan bahwa semua fitur berada dalam skala yang sama dan mencegah fitur dengan skala besar mendominasi fitur dengan skala kecil.

Linear Regression

Model regresi linier dengan menggunakan LinearRegression dari scikit-learn. Model ini kemudian dilatih dengan data latih menggunakan metode fit.

Prediction

Membuat prediksi pada set pengujian. Prediksi awal adalah nilai kontinu yang dihasilkan oleh model regresi linier. Prediksi ini kemudian dikonversi menjadi nilai biner berdasarkan threshold yang ditentukan (misalnya, 0.5).

# Model Evaluation

Confussion Matrix & ROC Curve

Menampilkan confusion matrix menggunakan confusion_matrix dan ConfusionMatrixDisplay dari scikit-learn untuk mengevaluasi performa model. ROC curve juga dibuat untuk menunjukkan trade-off antara true positive rate dan false positive rate. Area under the ROC curve (AUC) dihitung untuk memberikan satu nilai metrik dari performa model.

# Data Visualization

Membuat scatter plot untuk visualisasi hubungan antara dua fitur tertentu. hue digunakan untuk mewarnai titik-titik berdasarkan label biner (label_bnc) yang membantu dalam memahami distribusi data dan korelasi antar fitur.










