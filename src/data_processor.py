import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import bentoml

from omegaconf import DictConfig
import hydra
import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning)

def load_data(data_path: str) -> pd.DataFrame:
    data = pd.read_csv(data_path, sep='\t', encoding='latin1')

    return data

def drop_na(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()

def get_age(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(age=df['Year_Birth'].apply(lambda row: 2022 - row))

def get_total_childern(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(total_children= df['Kidhome'] + df['Teenhome'])

def get_total_purchases(df: pd.DataFrame) -> pd.DataFrame:
    purchases_columns = df.filter(like='Purchases', axis=1).columns

    return df.assign(total_purchases=df[purchases_columns].sum(axis=1))

def get_enrollment_years(df: pd.DataFrame) -> pd.DataFrame:
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], infer_datetime_format=True)
    return df.assign(enrollment_years=2022 - df["Dt_Customer"].dt.year)


def get_family_size(df: pd.DataFrame, size_map: dict) -> pd.DataFrame:
    return df.assign(
        family_size=df["Marital_Status"].map(size_map) + df["total_children"]
    )


def drop_features(df: pd.DataFrame, keep_columns: list):
    df = df[keep_columns]
    return df


def drop_outliers(df: pd.DataFrame, column_threshold: dict):
    for col, threshold in column_threshold.items():
        df = df[df[col] < threshold]
    return df.reset_index(drop=True)


def drop_columns_and_rows(df: pd.DataFrame, columns: DictConfig):
    df = df.pipe(drop_features, keep_columns=columns["keep"]).pipe(
        drop_outliers, column_threshold=columns["remove_outliers_threshold"]
    )

    return df

def get_scaler(df: pd.DataFrame):
    scaler = StandardScaler()
    scaler.fit(df)

    return scaler

def scale_features(df: pd.DataFrame, scaler: StandardScaler):
    return pd.DataFrame(scaler.transform(df), columns=df.columns)

def get_pca_model(data: pd.DataFrame) -> PCA:
    pca = PCA(n_components=3)
    pca.fit(data)

    return pca

def reduce_dimension(df: pd.DataFrame, pca: PCA) -> pd.DataFrame:
    return pd.DataFrame(pca.transform(df), columns=["col1", "col2", "col3"])



@hydra.main(config_path='../config', config_name='config')
def process_data(config: DictConfig):
    df = load_data(config.raw_data.path)
    df = drop_na(df)
    df = get_age(df)
    df = get_total_childern(df)
    df = get_total_purchases(df)
    df = get_enrollment_years(df)
    df = get_family_size(df, config.encode.family_size)
    df = drop_columns_and_rows(df, config.columns)

    if not os.path.isdir(config.intermediate.dir):
        os.mkdir(config.intermediate.dir)

    df.to_csv(config.intermediate.path, index=False)

    # Scale data
    scaler = get_scaler(df)
    bentoml.sklearn.save_model('scaler', scaler, signatures={"predict": {"batchable": True}},)

    df = scale_features(df, scaler)

    # Reduce dimesion
    pca = get_pca_model(df)
    bentoml.sklearn.save_model('pca', pca, signatures={"predict": {"batchable": True}},)

    pca_df = reduce_dimension(df, pca)

    # Save processed data
    if not os.path.isdir(config.final.dir):
        os.mkdir(config.final.dir)
    pca_df.to_csv(config.final.path, index=False)


if __name__ == '__main__':
    process_data()

