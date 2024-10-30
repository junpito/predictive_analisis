# Laporan Proyek Machine Learning - Junpito Salim

## Domain Proyek

Kesehatan hutan memegang peranan penting dalam menjaga keseimbangan ekosistem, pengaturan iklim, dan sebagai sumber daya vital bagi kehidupan manusia dan satwa liar. Namun, ancaman terhadap kesehatan hutan, seperti perubahan iklim, kebakaran hutan, dan aktivitas manusia, terus meningkat. Pemantauan dan identifikasi kondisi kesehatan hutan menjadi krusial agar langkah pencegahan dan mitigasi dapat diambil tepat waktu dan secara efektif. Saat ini, metode pemantauan secara manual sangat memakan waktu dan sumber daya, khususnya pada area hutan yang luas. Oleh karena itu, pengembangan sistem otomatis yang mampu mengidentifikasi kondisi kesehatan pohon secara cepat dan akurat menjadi solusi yang mendesak.

**Relevansi Masalah**:
Dalam rangka mempertahankan fungsi vital hutan, diperlukan teknologi yang mampu melakukan klasifikasi kesehatan hutan secara efektif dan efisien. Mengembangkan model machine learning untuk tugas ini akan mengurangi ketergantungan pada tenaga manual serta mempercepat deteksi perubahan kesehatan pada hutan.
  
**Referensi:** 
- [Evolution and Paradigm Shift in Forest Health Research: A Review on of Global Trends and Knowledge Gaps](https://www.mdpi.com/1999-4907/15/8/1279) 
- [Classification of single tree decay stages from combined airborne LiDAR data and CIR imagery](https://www.tandfonline.com/doi/epdf/10.1080/10095020.2024.2311861?needAccess=true)
## Business Understanding

### Problem Statements

1. Pemantauan kesehatan hutan secara manual sangat memakan waktu dan membutuhkan banyak sumber daya, terutama untuk area hutan yang luas.
2. Dibutuhkan sebuah sistem otomatis yang mampu mengidentifikasi dan mengklasifikasi kondisi kesehatan pohon secara cepat dan akurat untuk mengurangi risiko kerusakan ekosistem.

### Goals
1. Mengembangkan model machine learning yang mampu mengklasifikasikan status kesehatan pohon atau area hutan dengan menggunakan label kesehatan: 'Healthy', 'Very Healthy', 'Sub-healthy', atau 'Unhealthy'.
2. Menyediakan sistem klasifikasi yang dapat diimplementasikan secara praktis untuk mendeteksi perubahan kesehatan hutan lebih awal, yang kemudian bisa membantu tindakan pencegahan dan mitigasi.

### Solution statements
1. Membangun model klasifikasi dengan variabel target Health_Status (kesehatan hutan) untuk mengidentifikasi kategori kesehatan hutan.
2. Menggunakan beberapa algoritma machine learning seperti Neural Network, Random Forest, dan XGBoost, serta memilih model terbaik berdasarkan metrik evaluasi seperti F1-Score, Precision, dan Recall.

## Data Understanding
Dataset ini merupakan koleksi komprehensif dari berbagai pengukuran ekologi dan lingkungan yang berfokus pada karakteristik pohon serta kondisi lokasi tempat pohon berada. Setiap entri dalam dataset mewakili pohon atau plot tertentu, dengan berbagai fitur yang menyediakan gambaran lengkap mengenai kondisi ekosistem pada wilayah pengamatan. 
[Forest Health and Ecological Diversity](https://www.kaggle.com/datasets/ziya07/forest-health-and-ecological-diversity/data).

### Variabel-variabel pada Forest Health and Ecological Diversity dataset adalah sebagai berikut:
- `Plot_ID`: ID unik untuk setiap plot pengukuran, memudahkan identifikasi lokasi spesifik dalam area studi.
- `Latitude` & `Longitude`: Koordinat geografis plot dalam derajat, menunjukkan lokasi plot di permukaan bumi untuk posisi utara-selatan dan timur-barat.
- `DBH` (Diameter at Breast Height): Diameter pohon pada ketinggian dada (1,3 meter) diukur dalam sentimeter. Pengukuran ini penting dalam menilai ukuran dan kesehatan pohon.
- `Tree_Height`: Tinggi total pohon dari dasar hingga puncak, diukur dalam meter, yang membantu memahami pola pertumbuhan dan peran ekologis pohon.
- `Crown_Width_North_South` & `Crown_Width_East_West`: Lebar mahkota pohon diukur dalam arah utara-selatan dan timur-barat, biasanya dalam meter. Memberikan pandangan menyeluruh mengenai ukuran kanopi pohon.
- `Slope`: Kemiringan lahan pada lokasi pohon, diukur dalam derajat. Dapat mempengaruhi drainase air, erosi tanah, dan perkembangan akar.
- `Elevation`: Tinggi plot dari permukaan laut, diukur dalam meter. Elevasi memengaruhi suhu, curah hujan, dan dinamika ekosistem.
- `Temperature` & `Humidity`: Suhu rata-rata (°C) dan kelembapan (%) di plot, yang dapat mempengaruhi pertumbuhan dan distribusi spesies pohon.
- `Soil_TN` (Total Nitrogen) & `Soil_TP` (Total Phosphorus): Konsentrasi total nitrogen dan fosfor dalam tanah, diukur dalam gram per kilogram (g/kg). Nutrisi penting untuk pertumbuhan dan perkembangan tanaman.
- `Soil_AP` (Available Phosphorus) & `Soil_AN` (Available Nitrogen): Jumlah fosfor dan nitrogen yang tersedia bagi tanaman di tanah, juga diukur dalam g/kg. Menggambarkan tingkat kesuburan tanah.
- `Menhinick_Index` & `Gleason_Index`: Indeks keanekaragaman yang mencerminkan kekayaan spesies di area tersebut. Nilai yang lebih tinggi menunjukkan keanekaragaman hayati yang lebih besar.
- `Disturbance_Level`: Variabel kategori yang menunjukkan tingkat gangguan ekologi di area (0: rendah, 1: sedang, 2: tinggi), yang mempengaruhi kesehatan dan stabilitas ekosistem.
- `Fire_Risk_Index`: Ukuran potensi kebakaran berdasarkan kondisi lingkungan, dengan skala 0 hingga 1. Dapat menginformasikan strategi manajemen di area rawan kebakaran.
- `Health_Status`: Variabel kategori yang menunjukkan kesehatan pohon, diklasifikasikan sebagai ‘Healthy’ atau ‘Unhealthy’. Informasi ini krusial untuk memahami pengaruh faktor lingkungan terhadap vitalitas pohon.

## Data understanding & Explorasi data
Pada tahap ini, dilakukan eksplorasi data untuk memahami struktur dan karakteristik data. Proses ini bertujuan untuk mengidentifikasi pola dasar, anomali, dan informasi penting lain yang berguna untuk langkah pemrosesan selanjutnya dalam membangun model machine learning. Langkah-langkah memahami data tersebut meliputi:
1. **Meninjau Struktur Dataset:** Pada langkah ini kita akan melihat jumlah kolom, type data dan nilai yang terdapat pada dataset. Berikut adalah rangkuman informasi dari proses ini:
* Dataset terdiri dari 1000 baris dan 20 kolom.
* Seluruh nilai pada dataset berbentuk float kecuali pada kolok `Health_Status` berbentuk object.
* Data Kosong: Tidak ditemukan data kosong di seluruh kolom.
* Data Duplikat: Tidak ada data duplikasi dalam dataset.
* Menggunakan boxplot untuk mendeteksi outlier pada fitur numerik.
* Berdasarkan visualisasi boxplot, tidak ditemukan outlier pada seluruh variabel numerik.
2. **Analisis statistik sederhana:** Tahap deskripsi statistik dataset memberikan informasi mengenai distribusi dari setiap variabel numerik. berikut adalah rangkuman dari proses ini:
* Rata-rata diameter pohon pada ketinggian dada (DBH) adalah 52.73 cm, dengan rata-rata tinggi pohon 15.73 meter.
* Rata-rata indeks risiko kebakaran berada di 0.51, yang menunjukkan kemungkinan sedang untuk kebakaran.
* Terdapat variasi cukup besar di beberapa fitur, seperti DBH (Standar Deviasi = 27.61) dan Elevation (Standar Deviasi = 826.25), menunjukkan persebaran nilai yang luas dalam variabel tersebut.
* Fitur ekologis seperti Menhinick_Index dan Gleason_Index juga memiliki variasi yang signifikan.
* Untuk variabel Latitude dan Longitude, distribusi berkisar dari sekitar 10 hingga hampir 50 derajat latitude dan -129 hingga -60 derajat longitude.
* Elevation memiliki nilai minimum sekitar 100 meter dan maksimum 2172 meter, menandakan variasi tinggi geografis antar plot.
* Beberapa variabel memiliki nilai minimum mendekati nol, seperti Slope, Disturbance_Level, dan Fire_Risk_Index.

3. **Visualisasi:** Tahap ini dialakukan visualisasi untuk melihat sebaran data menggunakan fitur `Marker` dari librari `follium`, fitur `hist`dan `pairplot` dari librari `seaborn`. Berikut adalah kesimpulan dari tahapan ini:
* Titik kordinat tersebar di seluruh area Amerika.
* Sebagian besar variabel menunjukkan distribusi yang merata atau mendekati seragam, yang menunjukkan bahwa nilai dalam dataset tersebar hampir merata di seluruh rentang yang tersedia.
* Tidak ada korelasi linier yang kuat antara sebagian besar pasangan fitur.
* Data tersebar secara acak atau memiliki noise yang tinggi, yang berarti bahwa hubungan antar fitur tidak terlihat jelas dalam bentuk pola terstruktur.

## Data Preparation
Proses persiapan data ini bertujuan untuk mempersiapkan dataset agar sesuai dengan kebutuhan model machine learning, dengan fokus pada konversi data kategori, normalisasi, pembagian dataset, dan penanganan data imbalance. Tahapan yang dilakukan adalah sebagai berikut:

1. **Mapping Target ke Numerik**
Proses: Konversi nilai kategori pada kolom `Health_Status` ke format numerik. Kategori 'Healthy', 'Very Healthy', 'Unhealthy', dan 'Sub-healthy' masing-masing dimapping menjadi nilai 0, 1, 2, dan 3 menggunakan dictionary `health_status_mapping`. Nilai-nilai ini kemudian diterapkan pada dataset menggunakan `.map()`.
Alasan: Model machine learning tidak dapat bekerja langsung dengan data kategori; oleh karena itu, konversi ini diperlukan agar model dapat memproses nilai target dalam bentuk numerik. Selain itu, pendekatan ini memudahkan model untuk melakukan perhitungan dan menginterpretasi variabel target sebagai variabel klasifikasi.
2.  **Feature Scaling**
Proses: Melakukan standarisasi pada fitur numerik menggunakan `StandardScaler`, yang mengubah setiap fitur menjadi distribusi dengan mean 0 dan standar deviasi 1.
Alasan: Scaling atau standarisasi diperlukan untuk memastikan semua fitur berada dalam skala yang sama, yang penting dalam algoritma machine learning seperti neural networks, Decision tree, dan XGBoost. Jika fitur memiliki rentang nilai yang berbeda, fitur dengan rentang yang lebih besar akan mendominasi perhitungan jarak atau nilai gradien, yang dapat mengakibatkan bias dalam pelatihan model.
3. **Splitting Data**
Proses: Memisahkan data menjadi data training dan data testing menggunakan fungsi `train_test_split` dengan ukuran test sebesar 20% dari data total. `random_state=42` digunakan agar hasil pembagian data konsisten setiap kali kode dijalankan.
Alasan: Pembagian dataset menjadi data training dan testing adalah langkah penting untuk mengevaluasi performa model secara objektif. Data training digunakan untuk melatih model, sementara data testing digunakan untuk menguji model pada data baru yang tidak terlihat saat pelatihan, sehingga memungkinkan kita untuk menilai generalisasi model.
4. **Oversampling**
Proses: Mengatasi ketidakseimbangan kelas dalam dataset dengan teknik oversampling menggunakan `SMOTE` (Synthetic Minority Over-sampling Technique). Data training yang tidak seimbang diduplikasi untuk kelas minoritas hingga mencapai distribusi yang lebih seimbang.
Alasan: Ketidakseimbangan kelas dapat menyebabkan model machine learning bias terhadap kelas mayoritas. Dengan menggunakan SMOTE, kita dapat meningkatkan jumlah sampel kelas minoritas secara sintetis, yang membantu model untuk belajar secara lebih seimbang terhadap semua kelas dalam data training.

## Modeling
Pada bagian ini, kita akan membahas model yang digunakan untuk klasifikasi label Health_Status, yaitu **Neural Network**, **Random Forest**, dan **XGBoost**. Setiap model memiliki kelebihan dan kekurangan yang menjadi bahan pertimbangan untuk menentukan model terbaik. Berikut adalah detail dari setiap model, parameter yang digunakan, dan alasan pemilihan model.

### 1. Neural Network
Neural Network adalah model yang fleksibel dan mampu mempelajari hubungan non-linear antar fitur. Model ini cocok untuk data berukuran besar dengan kompleksitas tinggi.

Implementasi:
Arsitektur: Menggunakan model Sequential dengan tiga lapisan dense.
* Layer 1: 16 neuron, relu sebagai fungsi aktivasi.
* Layer 2: 32 neuron, regularisasi l2 untuk mencegah overfitting.
* Output Layer: 4 neuron dengan softmax untuk klasifikasi multi-kelas.
Kompilasi: Menggunakan optimizer adam dan SparseCategoricalCrossentropy sebagai fungsi loss.
* Training: Model dilatih selama 100 epoch dengan data `x_train_smote` dan `y_train_smote`.


Evaluasi:
* Akurasi pada data testing: 86.79%
* F1-Score: 0.88
* Precision: 0.89
* Recall: 0.89

Kelebihan & Kekurangan:
* Kelebihan: Mampu mempelajari pola kompleks dalam data dan fleksibel dalam penyesuaian arsitektur.
* Kekurangan: Membutuhkan waktu komputasi yang lebih tinggi, sensitif terhadap overfitting pada dataset kecil atau tidak seimbang.

### 2. Random Forest
Random Forest adalah ensemble model berbasis pohon keputusan yang memiliki keunggulan dalam mengurangi varians dan mencegah overfitting dengan menggabungkan hasil dari banyak pohon.

Implementasi:
* Hyperparameter Tuning: Menggunakan Grid Search untuk menemukan jumlah estimator optimal (`n_estimators=50`).
Model Training: Menggunakan RandomForestClassifier dengan jumlah estimator terbaik (50) dan dilatih pada data `x_train_smote` dan `y_train_smote`.

Evaluasi:
* Akurasi pada data testing: 100%
* F1-Score: 1.00
* Precision: 1.00
* Recall: 1.00

Kelebihan & Kekurangan:
* Kelebihan: Tahan terhadap overfitting, bekerja baik dengan dataset yang tidak seimbang, dan interpretabilitas yang lebih baik.
* Kekurangan: Rentan terhadap overfitting pada dataset yang memiliki banyak fitur.
### 3. XGBoost
XGBoost adalah model berbasis boosting yang mengoptimalkan loss function dengan pendekatan ensemble secara bertahap. XGBoost sangat efektif untuk menangani data yang tidak seimbang.

Implementasi:
Parameter Model:
* `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, dan `colsample_bytree=0.8`.
* Training: Dilatih menggunakan data `x_train_smote` dan `y_train_smote`.

Evaluasi:
* Akurasi pada data testing: 99%
* F1-Score: 1.00
* Precision: 0.99
* Recall: 0.99

Kelebihan & Kekurangan:
* Kelebihan: Efektif untuk menangani data besar, memiliki fitur regularisasi untuk mencegah overfitting, dan bekerja baik pada data tidak seimbang.
* Kekurangan: Membutuhkan tuning hyperparameter yang kompleks untuk mencapai performa optimal.


### Pemilihan Model Terbaik
Berdasarkan hasil evaluasi, model terbaik untuk klasifikasi Health_Status pada dataset ini adalah **Random Forest**. Model ini memiliki akurasi, F1-score, precision, dan recall sebesar 100% pada data testing, menunjukkan bahwa model ini mampu mengklasifikasikan data dengan baik.

Alasan Pemilihan:
* Akurasi dan F1-Score Tinggi: Hasil evaluasi menunjukkan performa yang sangat baik pada metrik-metrik penting.
* Robust Terhadap Overfitting: Dengan menggabungkan beberapa pohon keputusan, Random Forest dapat mengurangi overfitting dan menjaga stabilitas model.
* Efisiensi Waktu Komputasi: Dibandingkan dengan Neural Network, Random Forest lebih efisien dalam waktu komputasi, terutama setelah optimasi parameter.

## Evaluation
Pada bagian ini, kinerja model dievaluasi menggunakan metrik-metrik yang relevan untuk masalah klasifikasi multi-kelas ini, yaitu akurasi, precision, recall, dan F1-score. Pemilihan metrik ini disesuaikan dengan tujuan proyek untuk memberikan prediksi akurat pada setiap kategori Health_Status (Healthy, Very Healthy, Unhealthy, dan Sub-healthy) secara seimbang, sehingga hasil evaluasi ini dapat mencerminkan performa model secara menyeluruh.

* Akurasi (Accuracy): Akurasi mengukur proporsi prediksi yang benar dari keseluruhan prediksi yang dihasilkan. Akurasi dapat dirumuskan sebagai: `Akurasi = Jumlah Prediksi Benar / Total Prediksi`.
* Precision: Precision atau presisi mengukur proporsi prediksi positif yang benar dari seluruh prediksi positif yang dihasilkan oleh model. Precision berguna ketika fokus utama adalah menghindari prediksi positif yang salah. Rumus precision adalah: `Precision = True Positive (TP) / (True Positive (TP) + False Positive (FP)) `

* Recall: Recall atau sensitivitas mengukur kemampuan model untuk mendeteksi semua sampel yang benar-benar positif dari keseluruhan sampel positif yang ada. Metrik ini penting ketika prioritas adalah menangkap semua kasus positif, meskipun terdapat beberapa prediksi positif yang salah. Rumus recall adalah: `Recall = True Positive (TP) / (True Positive (TP) + False Negative (FN)) `.
* F1-Score: F1-score adalah rata-rata harmonik dari precision dan recall. Metrik ini mempertimbangkan baik precision maupun recall sehingga sesuai untuk kasus yang memerlukan keseimbangan antara kedua metrik tersebut. Rumus F1-score adalah: `F1 Score = 2 * (Precision * Recall) / (Precision + Recall)`

### Hasil Evaluasi Proyek

**Neural Network:**
* Akurasi: 86.79%
* Precision: 88.82%
* Recall: 88.5%
* F1-Score: 88.36%
* Model Neural Network menunjukkan performa yang baik, dengan akurasi, precision, dan recall yang cukup tinggi. Namun, model ini menunjukkan sedikit ketidakseimbangan pada kelas tertentu, yang terlihat dari hasil confusion matrix.

**Random Forest:**
* Akurasi: 100%
* Precision: 100%
* Recall: 100%
* F1-Score: 100%
* Model Random Forest menunjukkan performa yang sangat baik pada semua metrik evaluasi. Hal ini menunjukkan bahwa model dapat memprediksi setiap kelas dengan sempurna. 

**XGBoost:**
* Akurasi: 99%
* Precision: 99.01%
* Recall: 99%
* F1-Score: 99.00%
* Model XGBoost juga menunjukkan performa yang sangat baik dengan akurasi yang tinggi.

### Dampak Terhadap Business Understaning 
**Menjawab Problem Statement:**
Berdasarkan hasil evaluasi, model Random Forest dan XGBoost memiliki akurasi dan recall yang sangat tinggi, yang berarti keduanya dapat secara efektif mengidentifikasi kondisi kesehatan hutan sehingga model yang dikembangkan bisa digunakan untuk mengembangkan sistem deteksi kerusakan hutan otomatis.

**Apakah Mencapai Goals yang Diharapkan?**
Model machine learning yang dikemabangkan mampu mengklasifikasi status kesehatan hutan secara akurat, dengan kategori Healthy, Very Healthy, Sub-healthy, atau Unhealthy. Sehingga mendukung tindakan pencegahan yang lebih cepat dan efisien.






**---Ini adalah bagian akhir laporan---**


