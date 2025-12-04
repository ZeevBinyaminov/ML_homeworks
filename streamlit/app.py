import pathlib
import pickle

import streamlit as st
import pandas as pd
import plotly.express as px


TRAIN_PATH = "datasets/train_df.csv"
MODEL_PATH = "models/baseLR.pkl"

original_feauture_names = ['year', 'km_driven', 'mileage',
                           'max_power', 'torque', 'max_torque_rpm']
feature_names = [*original_feauture_names, 'year^2']


@st.cache_data
def load_train():
    df = pd.read_csv(TRAIN_PATH)
    return df


@st.cache_data
def plot_eda_figures(df):
    cat_columns = ['fuel', 'seller_type', 'transmission', 'owner']
    st.subheader("Распределение целевой переменной")
    fig = px.histogram(df, x='selling_price', nbins=50,
                       title='Распределение целевой переменной')
    st.plotly_chart(fig)

    st.subheader("Распределение категориальных признаков")
    for column in cat_columns:
        fig = px.bar(df, x=column, title=f'Распределение {column}')
        st.plotly_chart(fig)


def load_model_and_scaler():
    if not pathlib.Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Файл модели не найден {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
        return data['best_model'], data['StandardScaler']


def display_model_weights(model):
    st.subheader("Веса лучшей модели")
    coefficients = model.coef_.round()

    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }, index=feature_names)

    fig = px.bar(coef_df['Coefficient'], hover_name=coef_df['Feature'],
                 title=f"Коэффициенты лучшей модели",
                 labels={'index': 'Переменная', 'value': 'Коэффициент'})
    st.plotly_chart(fig)


def inference_model(model, scaler, input_data: pd.DataFrame):
    input_data_scaled = pd.DataFrame(
        scaler.transform(input_data[original_feauture_names]),
        columns=input_data.columns)

    input_data_scaled['year^2'] = input_data_scaled['year'] ** 2
    predictions = model.predict(input_data_scaled)
    return predictions


if __name__ == "__main__":
    st.title("Анализ данных о продаже автомобилей")
    train_df = load_train()
    model, scaler = load_model_and_scaler()

    eda_plots = st.checkbox("Показать EDA графики", value=False)
    model_coefs = st.checkbox("Показать коэффициенты модели", value=False)

    if eda_plots:
        plot_eda_figures(train_df)
    if model_coefs:
        display_model_weights(model)

    inference = st.file_uploader(
        "Загрузите CSV файл для предсказания", type=['csv'])
    if inference is not None:
        input_df = pd.read_csv(inference)
        st.subheader("Входные данные для предсказания")
        st.dataframe(input_df)

        predictions = inference_model(model, scaler, input_data=input_df)
        input_df['predicted_selling_price'] = predictions.round(2)

        st.subheader("Результаты предсказания")
        st.dataframe(input_df)
