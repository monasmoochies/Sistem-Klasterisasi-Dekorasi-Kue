import os
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# =====================================================
#  BAGIAN 1 — GENERATOR CSV OTOMATIS
# =====================================================

# Jika file CSV belum ada, buat otomatis
if not os.path.exists("cake_annotated.csv"):
    folder = "images"

    files = sorted([f for f in os.listdir(folder)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    data = {
        "file_name": files,
        "cream": ["yes"] * len(files),
        "fruits": ["no"] * len(files),
        "sprinkle_toppings": ["no"] * len(files)
    }

    df_generated = pd.DataFrame(data)
    df_generated.to_csv("cake_annotated.csv", index=False)
    print("🍰 CSV otomatis berhasil dibuat!")


# =====================================================
#  BAGIAN 2 — APLIKASI STREAMLIT
# =====================================================

df = pd.read_csv("cake_annotated.csv")

# ============================
# SIDEBAR
# ============================
st.sidebar.title("📌 Menu Navigasi")
menu = st.sidebar.radio(
    "Pilih Mode:",
    ["📊 Dataset", "🎯 Klasterisasi K-Means"]
)

# ============================
# JUDUL
# ============================
st.title("🎂 Sistem Klasterisasi Dekorasi Kue")
st.write("Menggunakan Metode **K-Means Clustering** Berbasis Streamlit")
st.write("---")

# ============================
# MODE 1 — DATASET
# ============================
if menu == "📊 Dataset":

    st.header("📁 Eksplorasi Dataset Cake")

    with st.expander("ℹ️ Penjelasan Dekorasi"):
        st.write("""
        **Cream** → topping lembut seperti whipped cream atau buttercream  
        **Fruits** → topping buah segar  
        **Sprinkles** → taburan warna-warni seperti confetti
        """)

    st.subheader("📘 Dataset Lengkap")
    st.dataframe(df)

    st.subheader("🖼 Contoh Gambar Cake")
    folder = "images"
    if os.path.isdir(folder):
        imgs = sorted(os.listdir(folder))
        for img in imgs:
            st.image(os.path.join(folder, img),
                     caption=img,
                     use_container_width=True)


# ============================
# MODE 2 — K-MEANS
# ============================
elif menu == "🎯 Klasterisasi K-Means":

    st.header("🎯 Klasterisasi Dekorasi Kue dengan K-Means")

    # Ubah yes/no → angka
    df_num = df.copy()
    df_num["cream"] = df_num["cream"].map({"yes": 1, "no": 0})
    df_num["fruits"] = df_num["fruits"].map({"yes": 1, "no": 0})
    df_num["sprinkle_toppings"] = df_num["sprinkle_toppings"].map({"yes": 1, "no": 0})

    # HITUNG CLUSTER OTOMATIS
    total_data = len(df_num)

    if total_data <= 1:
        st.error("❗ Data terlalu sedikit untuk clustering.")
    else:
        # maksimal 3 cluster
        n_clusters = min(3, total_data)

        st.info(f"🔢 Jumlah cluster otomatis: **{n_clusters}** (data = {total_data})")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df_num["cluster"] = kmeans.fit_predict(
            df_num[["cream", "fruits", "sprinkle_toppings"]]
        )

        st.subheader("📌 Hasil Klaster")
        st.dataframe(df_num)

        # PIE CHART
        st.subheader("📊 Visualisasi Topping (Pie Chart)")
        fig, ax = plt.subplots()

        labels = ["Cream", "Fruits", "Sprinkles"]
        values = [
            df["cream"].value_counts().get("yes", 0),
            df["fruits"].value_counts().get("yes", 0),
            df["sprinkle_toppings"].value_counts().get("yes", 0)
        ]

        ax.pie(values, labels=labels, autopct='%1.1f%%')
        st.pyplot(fig)

        st.success(f"🎉 K-Means selesai! Terbentuk **{n_clusters}** cluster.")
