import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import math
import seaborn as sns  # Import Seaborn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from io import BytesIO
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score  # Impor silhouette_score
from sklearn.metrics import davies_bouldin_score  # Impor davies_bouldin_score

st.set_page_config(
    page_title="Aplikasi CLUSTERING UMKM",
    layout="wide",
    initial_sidebar_state="auto",
)

# Fungsi untuk menginisialisasi session state
def init_session():
    return {}

# Fungsi untuk mendapatkan atau membuat session state
def get_session_state():
    session_state = st.session_state.get('_session_state')
    if session_state is None:
        session_state = init_session()
        st.session_state['_session_state'] = session_state
    return session_state

# Fungsi untuk menyimpan data yang diunggah ke session state
def save_uploaded_data(df):
    session_state = get_session_state()
    session_state['uploaded_data'] = df

# Fungsi untuk mendapatkan data yang diunggah dari session state
def get_uploaded_data():
    session_state = get_session_state()
    return session_state.get('uploaded_data', None)

# Fungsi untuk menyimpan pilihan kolom ke dalam session state
def save_selected_columns(selected_columns):
    session_state = get_session_state()
    session_state['selected_columns'] = selected_columns

# Fungsi untuk mendapatkan pilihan kolom dari session state
def get_selected_columns():
    session_state = get_session_state()
    return session_state.get('selected_columns', [])

# Fungsi untuk Unggah file excel atau csv
def read_data():
    upload_file = st.file_uploader("Unggah file (Excel atau CSV)", type=["xlsx", "xls", "csv"])
    if upload_file is not None:
        # Cek format file yang diunggah
        if upload_file.name.endswith('.csv'):
            df = pd.read_csv(upload_file)
        else:
            df = pd.read_excel(upload_file, engine='openpyxl')
        
        save_uploaded_data(df)
        st.success("Data berhasil diunggah!")
        
        if st.button("Tampilkan data"):
            upload_data = get_uploaded_data()
            if upload_data is not None:
                st.dataframe(upload_data)
            else:
                st.warning("Data tidak ditemukan!")


# Fungsi untuk menyimpan data hasil missing values ke dalam session state
def save_missing_values_data(df_filled):
    st.session_state['df_filled'] = df_filled

# Fungsi untuk mendapatkan data hasil missing values dari session state
def get_missing_values_data():
    return st.session_state.get('df_filled', None)

# Fungsi untuk mengisi nilai yang hilang dengan nilai rata-rata
def missing_values_mean(df, selected_columns):
    df_filled = df.copy()
    for selected_column in selected_columns:
        if selected_column in df_filled.columns:
            if pd.api.types.is_numeric_dtype(df_filled[selected_column]):
                column_mean = df_filled[selected_column].mean()
                df_filled[selected_column].fillna(column_mean, inplace=True)
            
    st.write("Data setelah Missing Values diisi dengan Mean:")
    st.dataframe(df_filled)

    # Menyimpan data hasil missing values ke dalam session state
    save_missing_values_data(df_filled)
    return df_filled

# Fungsi untuk menyimpan data hasil metode Z-Score ke dalam session state
def save_zscore_data(df_normalized):
    session_state = get_session_state()
    session_state['df_normalized'] = df_normalized

# Fungsi untuk mendapatkan data hasil metode Z-Score dari session state
def get_zscore_data():
    session_state = get_session_state()
    return session_state.get('df_normalized', None)


# Fungsi untuk metode Z-Score Normalisasi
def zscore_normalization(df, selected_columns):
    df_normalized = df.copy()
    zscore_scaler = StandardScaler()

    # Pisahkan semua kolom non-numerik agar tidak diikutkan dalam normalisasi
    non_numeric_columns = df_normalized.select_dtypes(exclude=['float64', 'int64']).columns
    df_non_numeric = df_normalized[non_numeric_columns]  # Simpan kolom non-numerik

    # Lakukan Z-Score Normalisasi hanya pada kolom numerik yang dipilih
    df_numeric = df_normalized[selected_columns].select_dtypes(include=['float64', 'int64'])
    if not df_numeric.empty:
        df_normalized[df_numeric.columns] = zscore_scaler.fit_transform(df_numeric)

    # Gabungkan kembali kolom non-numerik setelah normalisasi
    df_normalized[non_numeric_columns] = df_non_numeric

    # Tampilkan hasil normalisasi di Streamlit
    st.write("Data setelah Z-Score Normalisasi:")
    st.dataframe(df_normalized)
    
    # Simpan data yang sudah dinormalisasi ke session state
    save_zscore_data(df_normalized)

    return df_normalized


# Fungsi untuk menyimpan data hasil k-means ke dalam session state
def save_kmeans_data(df_clustering_kmeans):
    session_state = get_session_state()
    session_state['df_clustering_kmeans'] = df_clustering_kmeans


# Fungsi untuk mendapatkan data hasil k-means dari session state
def get_kmeans_data():
    session_state = get_session_state()
    return session_state.get('df_clustering_kmeans', None)


# Fungsi untuk K-Means Clustering
def kmeans_clustering(df, selected_columns, num_clusters):
    # Pastikan hanya kolom numerik yang digunakan untuk clustering
    df_clustering_kmeans = df[selected_columns].select_dtypes(include=['float64', 'int64']).copy()

    # Lakukan proses K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df_clustering_kmeans['Cluster'] = kmeans.fit_predict(df_clustering_kmeans)


    # Tambahkan 1 ke hasil clustering agar cluster dimulai dari 1
    df_clustering_kmeans['Cluster'] = df_clustering_kmeans['Cluster'] + 1

    # Tambahkan kolom Nama Batik ke hasil clustering (jika ada dalam dataframe awal)
    if 'Nama Batik' in df.columns:
        df_clustering_kmeans['Nama Batik'] = df['Nama Batik']

    # Simpan hasil clustering ke session state
    save_kmeans_data(df_clustering_kmeans)

    return df_clustering_kmeans


# Fungsi untuk k-means hasilnya kelompok UMKM
def kmeans_tabel_clusters(data_kmeans, df, num_clusters):
    for cluster_num in range(1, num_clusters + 1):
        st.write(f"*Cluster {cluster_num}*")
        result_data_kmeans = data_kmeans[data_kmeans['Cluster'] == cluster_num][["Nama Batik", "jumlah konsumen", "jumlah komplain konsumen", "kerjasama dengan mitra", "reward untuk pelanggan", "pelatihan karyawan pertahun", "terdapat branding produk", "kenaikan harga bahan baku", "pelatihan pemilik pertahun", "memiliki surat ijin usaha", "jumlah variasi motif batik", "menerapkan new normal", "aturan pembelian batik offline", "fasilitas pencegahan covid-19", "pegawai bersertifikat IT", "pendidikan pemilik",
                                    "mempunyai marketplace", "fasilitas pembayaran online", "SI pengelolaan batik sendiri", "media pemasaran online", "jumlah karyawan", "biaya produksi perbulan", "biaya tenaga kerja pertahun", "keuntungan pertahun"]]

        # Tambahkan kolom "Nama Batik" dari variabel data
        result_data_kmeans["Nama Batik"] = df[data_kmeans['Cluster'] == cluster_num]["Nama Batik"].values

        # Ubah urutan kolom sehingga "Nama Batik" berada di awal
        result_data_kmeans = result_data_kmeans[['Nama Batik'] + [col for col in result_data_kmeans if col != 'Nama Batik']]

        # Hitung jumlah UMKM dalam cluster
        num_umkm_in_cluster = len(result_data_kmeans)

        # Tampilkan jumlah UMKM di bawah tabel
        st.write(f"Jumlah UMKM dalam Cluster {cluster_num}: {num_umkm_in_cluster}")

        st.dataframe(result_data_kmeans)

def pembobotan_wp(df_clustering_kmeans, selected_columns, weights):
    # Normalisasi bobot
    total_weight = sum(weights)
    weights_normalisasi = [w / total_weight for w in weights]
    
    # Ubah nilai normalisasi menjadi persentase dalam bentuk string
    weights_normalisasi_persen = [f"{w * 100:.2f}%" for w in weights_normalisasi]
    
    # Membuat DataFrame untuk menampilkan bobot, normalisasi, dan persentase
    weights_table = pd.DataFrame({
        'Kriteria': selected_columns,
        'Bobot': weights,
        'Normalisasi Bobot': weights_normalisasi,
        'Persentase Normalisasi Bobot': weights_normalisasi_persen  # Kolom dengan format persentase
    })

    # Tampilkan tabel bobot dan normalisasi bobot menggunakan Streamlit
    st.write("*Tabel Bobot dan Normalisasi Bobot:*")
    st.dataframe(weights_table)

    # Pisahkan data berdasarkan Cluster
    cluster_ids = sorted(df_clustering_kmeans['Cluster'].unique())
    
    # Cek cluster yang ada
    st.write(f"Cluster yang ditemukan: {cluster_ids}")

    # Inisialisasi DataFrame untuk hasil WP
    df_wp_results = pd.DataFrame()

    # Proses setiap cluster
    for cluster_id in cluster_ids:
        st.write(f"*Cluster {cluster_id}*")  # Menampilkan dengan penomoran dari 1

        # Filter data berdasarkan cluster
        cluster_data = df_clustering_kmeans[df_clustering_kmeans['Cluster'] == cluster_id].copy()

        # Pisahkan kolom 'Nama Batik' karena tidak digunakan dalam perhitungan WP
        nama_batik = cluster_data['Nama Batik']
        cluster_data_numeric = cluster_data[selected_columns].select_dtypes(include=['float64', 'int64'])

        # Untuk menghindari hasil WP bernilai 0, tambahkan epsilon kecil pada data
        epsilon = 1e-6
        cluster_data_numeric = cluster_data_numeric.replace(0, epsilon)

        # Hitung WP Score dengan pembobotan
        wp_scores = (cluster_data_numeric ** weights_normalisasi).prod(axis=1)
        cluster_data['WP Score'] = wp_scores

        # Urutkan data berdasarkan WP Score dari yang terbesar hingga terkecil
        cluster_data_sorted = cluster_data.sort_values(by='WP Score', ascending=False)

        # Menggabungkan kolom "Nama Batik" kembali
        cluster_data_sorted['Nama Batik'] = nama_batik

        # Mengubah urutan kolom sehingga "Nama Batik" berada di awal
        cluster_data_sorted = cluster_data_sorted[['Nama Batik'] + ['WP Score'] + selected_columns]

        # Menyimpan hasil cluster yang diurutkan ke dalam DataFrame hasil akhir
        df_wp_results = pd.concat([df_wp_results, cluster_data_sorted])

        # Menampilkan hasil pembobotan WP untuk cluster ini
        st.write(f"*Hasil Pembobotan Weighted Product (WP) untuk Cluster {cluster_id}*")
        st.dataframe(cluster_data_sorted[['Nama Batik', 'WP Score'] + selected_columns])

        # Visualisasi peringkat WP Score untuk cluster ini
        st.write(f"*Visualisasi Peringkat WP Score Cluster {cluster_id}*")
        st.bar_chart(cluster_data_sorted[['WP Score']])

    return df_wp_results

def save_elbow_method_results(sse):
    session_state = get_session_state()
    session_state['elbow_method_results'] = sse

def get_elbow_method_results():
    session_state = get_session_state()
    return session_state.get('elbow_method_results', None)

# Fungsi untuk metode Elbow
def elbow_method(data, selected_columns):
    # Pisahkan kolom non-numerik
    data_numeric = data[selected_columns].select_dtypes(include=['float64', 'int64'])
    
    sse = []
    k_range = range(1, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_numeric)
        sse.append(kmeans.inertia_)
    
    # Plotting the Elbow Graph
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, sse, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('SSE (Sum of Squared Errors)')
    plt.title('Elbow Method for Optimal k')
    
    return sse

def main():

    #pilih  menu
    menu=("Aplikasi Pengelompokan UMKM",["Upload Data","Preprocessing","Clustering","Evaluasi","Hasil K-Means", "Pembobotan WP"])

    # Tampilkan judul "Sistem Pengelompokan" di bagian atas kiri
    st.sidebar.title(menu[0])

    # Create a multiselect menu
    options = ["Upload Data","Preprocessing","Clustering","Evaluasi","Hasil K-Means", "Pembobotan WP"]
    selected_options = st.multiselect("Pilih Menu : ", options)

    # #tampilan menu
    # with st.sidebar.expander("Pilih Menu :"):
    #     select_menu = st.selectbox("Pilih Menu : ", menu[1])

    if "Upload Data" in selected_options:
    # if select_menu == "Upload Data":
        st.header("Aplikasi Pengelompokan UMKM menggunakan optimasi Elbow")
        st.write('\n')
        st.subheader("Silahkan unggah Data!")
        df = read_data()
        return df

    elif "Preprocessing" in selected_options:
        with st.sidebar.expander("Pilih Menu :"):
            select_submenu_preprocessing = st.selectbox("Pilih SubMenu :", ["Missing Values (Mean)","Normalisasi ZScore"] )
        # select_submenu_preprocessing = st.sidebar.selectbox("Pilih SubMenu : ", submenu_preprocessing)
        st.title("Data Preprocessing")

        if select_submenu_preprocessing == "Missing Values (Mean)":
            st.subheader("Missing Values/Imputasi Data (Mean)")
            df = get_uploaded_data()  # Mendapatkan data yang diunggah dari session state

            if df is not None:
                selected_columns = ["Nama Batik","jumlah konsumen", "jumlah komplain konsumen", "kerjasama dengan mitra", "reward untuk pelanggan", "pelatihan karyawan pertahun", "terdapat branding produk", "kenaikan harga bahan baku", "pelatihan pemilik pertahun", "memiliki surat ijin usaha", "jumlah variasi motif batik", "menerapkan new normal", "aturan pembelian batik offline", "fasilitas pencegahan covid-19", "pegawai bersertifikat IT", "pendidikan pemilik", 
                                    "mempunyai marketplace", "fasilitas pembayaran online", "SI pengelolaan batik sendiri", "media pemasaran online", "jumlah karyawan", "biaya produksi perbulan", "biaya tenaga kerja pertahun", "keuntungan pertahun"]
                if selected_columns:
                    df_filled = missing_values_mean(df, selected_columns)
            else:
                st.warning("Belum ada data yang diunggah")


        elif select_submenu_preprocessing == "Normalisasi ZScore":
            st.subheader("Halaman Normalisasi Zscore")
            df_filled = get_missing_values_data()  
            if  df_filled is not None:
                selected_columns = ["Nama Batik","jumlah konsumen", "jumlah komplain konsumen", "kerjasama dengan mitra", "reward untuk pelanggan", "pelatihan karyawan pertahun", "terdapat branding produk", "kenaikan harga bahan baku", "pelatihan pemilik pertahun", "memiliki surat ijin usaha", "jumlah variasi motif batik", "menerapkan new normal", "aturan pembelian batik offline", "fasilitas pencegahan covid-19", "pegawai bersertifikat IT", "pendidikan pemilik", 
                                    "mempunyai marketplace", "fasilitas pembayaran online", "SI pengelolaan batik sendiri", "media pemasaran online", "jumlah karyawan", "biaya produksi perbulan", "biaya tenaga kerja pertahun", "keuntungan pertahun"]
                # Melakukan normalisasi pada semua kolom yang dipilih
                df_normalized = zscore_normalization(df_filled[selected_columns], selected_columns)
                
            else:
                st.warning("Belum ada data yang dilabel encoding. Silakan lakukan label encoding terlebih dahulu.")

    
    elif "Clustering" in selected_options:
        st.header("K-Means Clustering")
        df_normalized = get_zscore_data()

        # Periksa apakah data sudah diunggah dan tersedia
        if df_normalized is not None:
            selected_columns = ["Nama Batik", "jumlah konsumen", "jumlah komplain konsumen", "kerjasama dengan mitra", 
                                "reward untuk pelanggan", "pelatihan karyawan pertahun", "terdapat branding produk", 
                                "kenaikan harga bahan baku", "pelatihan pemilik pertahun", "memiliki surat ijin usaha", 
                                "jumlah variasi motif batik", "menerapkan new normal", "aturan pembelian batik offline", 
                                "fasilitas pencegahan covid-19", "pegawai bersertifikat IT", "pendidikan pemilik", 
                                "mempunyai marketplace", "fasilitas pembayaran online", "SI pengelolaan batik sendiri", 
                                "media pemasaran online", "jumlah karyawan", "biaya produksi perbulan", 
                                "biaya tenaga kerja pertahun", "keuntungan pertahun"]
            
            if selected_columns:
                # Tampilkan grafik Elbow Method hanya sekali
                st.subheader("Grafik Elbow Method untuk K-Means")
                sse = get_elbow_method_results()
                if sse is None:
                    sse = elbow_method(df_normalized, selected_columns)

                if sse:  # Pastikan SSE tidak kosong
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(1, 11), sse, 'bx-')
                    plt.xlabel('Jumlah cluster')
                    plt.ylabel('Sum of Square Error (SSE)')
                    plt.title('Metode Elbow untuk Menentukan Jumlah Cluster Optimal')

                    # Menemukan titik elbow menggunakan KneeLocator
                    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
                    optimal_k = kl.elbow

                    # Tambahkan titik lingkaran pada titik elbow jika ditemukan
                    if optimal_k is not None:
                        plt.scatter(optimal_k, sse[optimal_k-1], s=300, c='red', marker='o', label='Optimal Cluster')
                        plt.legend()
                    else:
                        st.warning("Tidak ditemukan titik elbow yang jelas.")

                    st.pyplot(plt)

                # Input untuk memilih jumlah cluster dan menjalankan clustering
                num_clusters = st.number_input('Pilih jumlah kluster:', min_value=1, max_value=10, value=2)
                st.session_state['num_clusters'] = num_clusters

                df_clustering_kmeans = kmeans_clustering(df_normalized, selected_columns, num_clusters)
                st.write("Data setelah K-Means Clustering:")
                st.dataframe(df_clustering_kmeans)
        else:
            st.warning("Belum ada data yang diunggah.")



    elif "Evaluasi" in selected_options:
        st.header("Halaman Evaluasi")

        df_clustering_kmeans = get_kmeans_data()

        silhouette_scores = []
        sse_scores = []

        if 'optimal_clusters_silhouette' not in st.session_state:
            st.session_state.optimal_clusters_silhouette = None
        if 'optimal_silhouette_score' not in st.session_state:
            st.session_state.optimal_silhouette_score = None

        # Pastikan data sudah tersedia
        if df_clustering_kmeans is not None and not df_clustering_kmeans.empty:
            # Hanya memilih kolom numerik untuk clustering
            selected_columns_numeric = df_clustering_kmeans.select_dtypes(include=['float64', 'int64']).columns.tolist()

            if selected_columns_numeric:
                # Loop melalui jumlah cluster dari 2 hingga 10
                for n_clusters in range(2, 11):
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                    cluster_labels = kmeans.fit_predict(df_clustering_kmeans[selected_columns_numeric])

                    # Hitung Silhouette Score
                    silhouette_avg = silhouette_score(df_clustering_kmeans[selected_columns_numeric], cluster_labels)
                    silhouette_scores.append(silhouette_avg)

                    # Hitung SSE
                    sse = kmeans.inertia_
                    sse_scores.append(sse)

                # Menampilkan hasil evaluasi dalam bentuk DataFrame
                results_df = pd.DataFrame({
                    'Jumlah Cluster': range(2, 11),
                    'Silhouette Score': silhouette_scores,
                    'SSE': sse_scores  # Tambahkan kolom SSE
                })

                # Menampilkan hasil dalam tabel
                st.write("Hasil Evaluasi K-Means Clustering:")
                st.write(results_df)

                # Membuat grafik Silhouette Score
                st.subheader("Grafik Silhouette Score")
                st.line_chart(results_df[['Jumlah Cluster', 'Silhouette Score']].set_index('Jumlah Cluster'))

                # Membuat grafik SSE
                st.subheader("Grafik SSE (Sum of Squared Errors)")
                st.line_chart(results_df[['Jumlah Cluster', 'SSE']].set_index('Jumlah Cluster'))

                # Menemukan cluster paling optimal berdasarkan Silhouette Score
                optimal_idx_silhouette = results_df['Silhouette Score'].idxmax()
                st.session_state.optimal_clusters_silhouette = results_df.loc[optimal_idx_silhouette]['Jumlah Cluster']
                st.session_state.optimal_silhouette_score = results_df.loc[optimal_idx_silhouette]['Silhouette Score']

                st.write(f"Cluster paling optimal berdasarkan Silhouette Score: {st.session_state.optimal_clusters_silhouette}")
                st.write(f"Silhouette Score terbaik: {st.session_state.optimal_silhouette_score}")

            else:
                st.warning("Tidak ada kolom numerik yang ditemukan dalam data. Pastikan data memiliki kolom numerik untuk dilakukan clustering.")

        else:
            st.warning("Belum ada data yang diunggah.")

    elif "Hasil K-Means" in selected_options:
        submenu_clustering = st.sidebar.selectbox("Pilih Metode Clustering:", ["Hasil K-Means"])
        st.subheader("Halaman Hasil K-Means")
        
        # Ambil data asli yang diunggah
        data_asli = get_uploaded_data()

        # Mengambil num_clusters dari session state
        num_clusters = st.session_state.get('num_clusters', 3)

        if data_asli is not None:
            if submenu_clustering == "Hasil K-Means":
                data_kmeans = get_kmeans_data()
                if data_kmeans is not None:
                    st.write("*Hasil K-Means Clustering*")

                    # Tambahkan kolom 'Cluster' ke data asli
                    data_asli['Cluster'] = data_kmeans['Cluster']

                    # Menampilkan informasi dan data setiap cluster secara terpisah
                    for cluster_num in range(1, num_clusters + 1):
                        st.write(f"*Cluster {cluster_num}*")

                        # Filter data untuk cluster tertentu
                        cluster_data = data_asli[data_asli['Cluster'] == cluster_num]

                        # Tampilkan data untuk cluster tertentu
                        st.dataframe(cluster_data)

                        # Tampilkan jumlah UMKM dalam cluster
                        st.write(f"Jumlah UMKM dalam Cluster {cluster_num}: {len(cluster_data)}")

                        # Pisahkan dengan garis horizontal
                        st.write('---')

                    # Membuat DataFrame untuk menyimpan hasil rata-rata per cluster
                    cluster_means = pd.DataFrame()

                    # Hitung rata-rata per variabel untuk setiap cluster
                    for cluster_num in range(1, num_clusters + 1):
                        cluster_data = data_asli[data_asli['Cluster'] == cluster_num]

                        # Hitung rata-rata per variabel untuk cluster ini
                        cluster_mean = cluster_data.mean(numeric_only=True)

                        # Menambahkan hasil rata-rata ke DataFrame
                        cluster_means[f"Cluster {cluster_num}"] = cluster_mean

                    # Tampilkan tabel rata-rata per cluster
                    st.write("Tabel Rata-rata Per Variabel untuk Setiap Cluster:")
                    st.dataframe(cluster_means.T)  # Transpose agar cluster jadi kolom
                else:
                    st.warning("Data hasil K-Means tidak ditemukan. Lakukan clustering terlebih dahulu.")
        else:
            st.warning("Data asli tidak ditemukan. Harap unggah data terlebih dahulu.")

    # Bagian aplikasi Streamlit untuk menampilkan pembobotan WP
    elif "Pembobotan WP" in selected_options:
        st.subheader("Halaman Hasil Pembobotan Weighted Product")

        df_clustering_kmeans = get_kmeans_data()
        if df_clustering_kmeans is not None:
            st.write("*Perhitungan Weighted Product (WP)*")

            # Ambil kolom yang relevan untuk perhitungan WP
            selected_columns = ["jumlah konsumen", "jumlah komplain konsumen", "kerjasama dengan mitra", 
                            "reward untuk pelanggan", "pelatihan karyawan pertahun", "terdapat branding produk", 
                            "kenaikan harga bahan baku", "pelatihan pemilik pertahun", "memiliki surat ijin usaha", 
                            "jumlah variasi motif batik", "menerapkan new normal", "aturan pembelian batik offline", 
                            "fasilitas pencegahan covid-19", "pegawai bersertifikat IT", "pendidikan pemilik", 
                            "mempunyai marketplace", "fasilitas pembayaran online", "SI pengelolaan batik sendiri", 
                            "media pemasaran online", "jumlah karyawan", "biaya produksi perbulan", 
                            "biaya tenaga kerja pertahun", "keuntungan pertahun"]

            # Tampilkan kolom yang digunakan dalam pembobotan WP
            st.write("Variabel Data UMKM Batik yang digunakan pada Pembobotan WP:")
            st.dataframe(pd.DataFrame(selected_columns, columns=["Kolom yang Relevan"]))

            # Asumsikan bobot untuk setiap kriteria
            weights = [4, 2, 4, 3, 2, 3, 1, 3, 2, 3, 1, 
                    1, 1, 4, 2, 3, 3, 2, 4, 4, 2, 2, 4]

            # Jalankan perhitungan pembobotan WP dan tampilkan hasilnya
            df_wp_results = pembobotan_wp(df_clustering_kmeans, selected_columns, weights)

        else:
            st.warning("Data tidak ada!")

   
if __name__ == "__main__":
    main()