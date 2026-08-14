import numpy as np
import pandas as pd
import streamlit as st
import torch

st.set_page_config(
    page_title="ガウス過程回帰によるシミュレーション",
    layout="wide",
)
st.title("ガウス過程回帰によるシミュレーション")

file = st.file_uploader("ファイルをアップロードしてください", type=["xlsx"])

if file is not None:
    st.write(f"{file.name} をアップロードしました！")

    excel_file = pd.ExcelFile(file)

    sheet_name = st.selectbox(
        "使用するシートを選択してください",
        excel_file.sheet_names,
    )
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    st.subheader("読み込んだデータ")
    st.dataframe(df, use_container_width=True)

    numeric_col = (df.select_dtypes(include=np.number).columns.tolist())

    st.header("入力変数・目的変数")

    col1, col2 = st.columns(2)

    with col1:

        x_columns = st.multiselect("入力変数 X", options=numeric_col)

    with col2:

        available_y_columns = [col for col in numeric_col if col not in x_columns]

        y_columns = st.multiselect("目的変数 Y", options=available_y_columns)

    if len(x_columns) == 0:
        st.info("入力変数を1つ以上選択してください")
        st.stop()

    if len(y_columns) == 0:
        st.info("目的変数を1つ以上選択してください")
        st.stop()

