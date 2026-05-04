"""Feature set and sklearn preprocessing utilities."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import TARGET_COLUMN

FEATURE_FAMILY_MAP = {
    "purchase_year": "order timing",
    "purchase_month": "order timing",
    "purchase_day": "order timing",
    "purchase_dayofweek": "order timing",
    "purchase_hour": "order timing",
    "weekend_purchase_flag": "order timing",
    "approval_delay_hours": "order timing",
    "approval_delay_days": "order timing",
    "estimated_delivery_days": "order timing",
    "n_items": "basket composition",
    "total_price": "basket composition",
    "avg_price": "basket composition",
    "max_price": "basket composition",
    "total_freight": "basket composition",
    "avg_freight": "basket composition",
    "freight_ratio": "basket composition",
    "n_distinct_products": "basket composition",
    "n_distinct_sellers": "basket composition",
    "total_basket_value": "basket composition",
    "freight_share": "basket composition",
    "avg_item_price": "basket composition",
    "expensive_order_flag": "basket composition",
    "multi_seller_flag": "basket composition",
    "multi_item_flag": "basket composition",
    "multi_product_flag": "basket composition",
    "n_payment_methods": "payment behavior",
    "n_payment_rows": "payment behavior",
    "max_payment_installments": "payment behavior",
    "total_payment_installments": "payment behavior",
    "total_payment_value": "payment behavior",
    "has_credit_card": "payment behavior",
    "has_boleto": "payment behavior",
    "has_voucher": "payment behavior",
    "has_debit_card": "payment behavior",
    "payment_value_diff": "payment behavior",
    "payment_count": "payment behavior",
    "multiple_payment_methods_flag": "payment behavior",
    "installment_purchase_flag": "payment behavior",
    "high_installment_flag": "payment behavior",
    "customer_state": "customer geography",
    "customer_city_freq": "customer geography",
    "n_seller_states": "seller geography",
    "primary_seller_state": "seller geography",
    "seller_customer_same_state": "seller geography",
    "same_state_flag": "seller geography",
    "multi_seller_state_flag": "seller geography",
    "n_categories": "product/category signals",
    "primary_category": "product/category signals",
    "avg_product_weight_g": "product/category signals",
    "max_product_weight_g": "product/category signals",
    "avg_product_photos_qty": "product/category signals",
    "avg_product_name_length": "product/category signals",
    "avg_product_description_length": "product/category signals",
    "total_product_volume_cm3": "product/category signals",
    "avg_product_volume_cm3": "product/category signals",
    "multi_category_flag": "product/category signals",
    "avg_product_volume_cm3_safe": "product/category signals",
    "total_product_volume_cm3_safe": "product/category signals",
    "estimated_days_per_item": "safe interactions",
    "freight_per_item": "safe interactions",
    "price_per_kg": "safe interactions",
}

REDUNDANT_FEATURE_COLUMNS = [
    "avg_price",
    "freight_ratio",
    "n_payment_rows",
    "seller_customer_same_state",
    "avg_product_volume_cm3",
    "total_product_volume_cm3",
    "freight_per_item",
]

PRIMARY_FEATURE_COLUMNS = [
    column for column in FEATURE_FAMILY_MAP
    if column not in REDUNDANT_FEATURE_COLUMNS
]

FORBIDDEN_LEAKAGE_COLUMNS = [
    "review_score",
    "review_comment_title",
    "review_comment_message",
    "review_creation_date",
    "review_answer_timestamp",
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "actual_lateness",
    "delivery_lateness",
]


def split_features_target(modeling_df):
    X = modeling_df.drop(columns=[TARGET_COLUMN])
    y = modeling_df[TARGET_COLUMN].copy()
    drop_columns = sorted(set(X.columns).intersection(FORBIDDEN_LEAKAGE_COLUMNS))
    if drop_columns:
        X = X.drop(columns=drop_columns)
    return X, y


def identify_feature_types(X):
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    unassigned = sorted(set(X.columns) - set(numeric_features) - set(categorical_features))
    if unassigned:
        raise ValueError(f"Unassigned feature columns: {unassigned}")
    return numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    try:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", one_hot_encoder),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

