
# 以下を「app.py」に書き込み
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import predict
import pandas as pd


# =========================
# ページ設定
# =========================
st.markdown(
    """
    <div style="text-align: center;">
        <h1>画像認識アプリ</h1>
        <p style="font-size:18px; color: #666;">
            画像を入力するとAIが何の画像かを判定します
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# =========================
# サイドバー
# =========================
st.sidebar.title("入力設定")

img_source = st.sidebar.radio(
    "・画像のソース",
    ("画像をアップロード", "カメラで撮影")
)

if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader(
        "・画像ファイル",
        type=["png", "jpg", "jpeg"]
    )
else:
    img_file = st.sidebar.camera_input("カメラで撮影")


# =========================
# メイン画面
# =========================

# 画像が入力されたら処理
if img_file is not None:
    with st.spinner("推定中..."):
        img = Image.open(img_file)
        results = predict(img)

    # 上位結果
    n_top = 3
    top = results[0]

    # =========================
    # 強調表示（最上位）
    # =========================
    st.metric(
        label="最も可能性が高い判定結果",
        value=top[0],
        delta=f"{round(top[2] * 100, 2)} %"
    )

    # 信頼度バー
    st.progress(int(top[2] * 100))
    st.caption("予測の信頼度")

    st.divider()

    # =========================
    # 画像 + 結果（2カラム）
    # =========================
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("入力画像")
        st.image(img, use_container_width=True)

    with col2:
        rank_icons = ["🥇", "🥈", "🥉"]

        st.subheader("判定結果（上位3位）")

        for i, result in enumerate(results[:3]):
            label = result[0]
            prob = round(result[2] * 100, 2)

            st.write(
                f"{rank_icons[i]} **{i+1}位：{label}**　{prob} %"
            )



    # =========================
    # グラフ + CSV（2カラム）
    # =========================
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("グラフ")

        pie_labels = [result[1] for result in results[:n_top]]
        pie_labels.append("others")
        pie_probs = [result[2] for result in results[:n_top]]
        pie_probs.append(sum([result[2] for result in results[n_top:]]))
        fig, ax = plt.subplots()
        wedgeprops={"width":0.3, "edgecolor":"white"}
        textprops = {"fontsize":6}
        ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90,
                textprops=textprops, autopct="%.2f", wedgeprops=wedgeprops)  # 円グラフ
        st.pyplot(fig)

    with col2:
        st.subheader("CSVダウンロード")
        df = pd.DataFrame({
        "ラベル": [r[0] for r in results],
        "確率(%)": [r[2] * 100 for r in results]
        })

        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "結果をCSVでダウンロード",
            csv,
            "prediction_result.csv",
            "text/csv"
        )

else:
    st.info("サイドバーから画像を入力してください")
