"""Feature engineering copied from the stable notebook pipeline."""

import numpy as np
import pandas as pd

from .config import LOW_REVIEW_THRESHOLD, TARGET_COLUMN
from .utils import merge_order_level, require_unique_key


def first_mode(series: pd.Series):
    modes = series.dropna().mode()
    return modes.iloc[0] if not modes.empty else np.nan


def deduplicate_reviews(order_reviews: pd.DataFrame) -> pd.DataFrame:
    reviews = (
        order_reviews.groupby("order_id", as_index=False)
        .agg(
            review_score=("review_score", "min"),
            n_review_rows=("review_id", "size"),
            n_distinct_review_ids=("review_id", "nunique"),
        )
    )
    reviews[TARGET_COLUMN] = np.select(
        [reviews["review_score"] <= LOW_REVIEW_THRESHOLD, reviews["review_score"] >= LOW_REVIEW_THRESHOLD + 1],
        [1, 0],
        default=np.nan,
    )
    if reviews[TARGET_COLUMN].isna().any():
        raise ValueError("Unexpected review_score values found outside the target definition.")
    reviews[TARGET_COLUMN] = reviews[TARGET_COLUMN].astype(int)
    require_unique_key(reviews, "order_id", "reviews_deduped")
    return reviews


def build_orders_base(orders: pd.DataFrame, reviews_deduped: pd.DataFrame) -> pd.DataFrame:
    orders_delivered = orders.loc[orders["order_status"] == "delivered"].copy()
    target_columns = ["order_id", "review_score", TARGET_COLUMN, "n_review_rows", "n_distinct_review_ids"]
    orders_base = orders_delivered.merge(
        reviews_deduped[target_columns],
        on="order_id",
        how="inner",
        validate="one_to_one",
    )
    return orders_base.drop(columns=["review_score", "n_review_rows", "n_distinct_review_ids"])


def aggregate_order_items(order_items: pd.DataFrame) -> pd.DataFrame:
    items_agg = (
        order_items.groupby("order_id", as_index=False)
        .agg(
            n_items=("order_item_id", "count"),
            total_price=("price", "sum"),
            avg_price=("price", "mean"),
            max_price=("price", "max"),
            total_freight=("freight_value", "sum"),
            avg_freight=("freight_value", "mean"),
            n_distinct_products=("product_id", "nunique"),
            n_distinct_sellers=("seller_id", "nunique"),
        )
    )
    items_agg["freight_ratio"] = np.where(
        items_agg["total_price"] > 0,
        items_agg["total_freight"] / items_agg["total_price"],
        np.nan,
    )
    return items_agg


def aggregate_product_features(order_items: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    product_columns = [
        "product_id",
        "product_category_name_english",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    items_products = order_items.merge(
        products[product_columns],
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    items_products["product_volume_cm3"] = (
        items_products["product_length_cm"]
        * items_products["product_height_cm"]
        * items_products["product_width_cm"]
    )
    return (
        items_products.groupby("order_id", as_index=False)
        .agg(
            n_categories=("product_category_name_english", "nunique"),
            primary_category=("product_category_name_english", first_mode),
            avg_product_weight_g=("product_weight_g", "mean"),
            max_product_weight_g=("product_weight_g", "max"),
            avg_product_photos_qty=("product_photos_qty", "mean"),
            avg_product_name_length=("product_name_lenght", "mean"),
            avg_product_description_length=("product_description_lenght", "mean"),
            total_product_volume_cm3=("product_volume_cm3", "sum"),
            avg_product_volume_cm3=("product_volume_cm3", "mean"),
        )
    )


def aggregate_payments(order_payments: pd.DataFrame, items_agg: pd.DataFrame) -> pd.DataFrame:
    payments_agg = (
        order_payments.groupby("order_id", as_index=False)
        .agg(
            n_payment_methods=("payment_type", "nunique"),
            n_payment_rows=("payment_sequential", "count"),
            max_payment_installments=("payment_installments", "max"),
            total_payment_installments=("payment_installments", "sum"),
            total_payment_value=("payment_value", "sum"),
            has_credit_card=("payment_type", lambda values: int("credit_card" in set(values))),
            has_boleto=("payment_type", lambda values: int("boleto" in set(values))),
            has_voucher=("payment_type", lambda values: int("voucher" in set(values))),
            has_debit_card=("payment_type", lambda values: int("debit_card" in set(values))),
        )
    )
    payments_agg = payments_agg.merge(
        items_agg[["order_id", "total_price", "total_freight"]],
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    payments_agg["payment_value_diff"] = payments_agg["total_payment_value"] - (
        payments_agg["total_price"] + payments_agg["total_freight"]
    )
    return payments_agg.drop(columns=["total_price", "total_freight"])


def aggregate_seller_features(order_items: pd.DataFrame, sellers: pd.DataFrame) -> pd.DataFrame:
    items_sellers = order_items.merge(
        sellers[["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"]],
        on="seller_id",
        how="left",
        validate="many_to_one",
    )
    return (
        items_sellers.groupby("order_id", as_index=False)
        .agg(
            n_seller_states=("seller_state", "nunique"),
            primary_seller_state=("seller_state", first_mode),
            primary_seller_city=("seller_city", first_mode),
            primary_seller_zip_code_prefix=("seller_zip_code_prefix", first_mode),
        )
    )


def add_engineered_features(model_df: pd.DataFrame) -> pd.DataFrame:
    features_df = model_df.copy()
    features_df["purchase_year"] = features_df["order_purchase_timestamp"].dt.year
    features_df["purchase_month"] = features_df["order_purchase_timestamp"].dt.month
    features_df["purchase_day"] = features_df["order_purchase_timestamp"].dt.day
    features_df["purchase_dayofweek"] = features_df["order_purchase_timestamp"].dt.dayofweek
    features_df["purchase_hour"] = features_df["order_purchase_timestamp"].dt.hour
    features_df["weekend_purchase_flag"] = features_df["purchase_dayofweek"].isin([5, 6]).astype(int)
    features_df["approval_delay_hours"] = (
        (features_df["order_approved_at"] - features_df["order_purchase_timestamp"]).dt.total_seconds() / 3600
    )
    features_df["approval_delay_days"] = features_df["approval_delay_hours"] / 24
    features_df["estimated_delivery_days"] = (
        (features_df["order_estimated_delivery_date"] - features_df["order_purchase_timestamp"]).dt.total_seconds() / 86400
    )
    features_df["total_basket_value"] = features_df["total_price"] + features_df["total_freight"]
    features_df["freight_share"] = np.where(
        features_df["total_basket_value"] > 0,
        features_df["total_freight"] / features_df["total_basket_value"],
        np.nan,
    )
    features_df["avg_item_price"] = features_df["avg_price"]
    expensive_order_threshold = features_df["total_basket_value"].quantile(0.75)
    features_df["expensive_order_flag"] = (features_df["total_basket_value"] >= expensive_order_threshold).astype(int)
    features_df["multi_seller_flag"] = (features_df["n_distinct_sellers"] > 1).astype(int)
    features_df["multi_item_flag"] = (features_df["n_items"] > 1).astype(int)
    features_df["multi_product_flag"] = (features_df["n_distinct_products"] > 1).astype(int)
    features_df["payment_count"] = features_df["n_payment_rows"]
    features_df["multiple_payment_methods_flag"] = (features_df["n_payment_methods"] > 1).astype(int)
    features_df["installment_purchase_flag"] = (features_df["max_payment_installments"] > 1).astype(int)
    features_df["high_installment_flag"] = (features_df["max_payment_installments"] >= 6).astype(int)
    customer_city_frequency = features_df["customer_city"].value_counts(normalize=True)
    features_df["customer_city_freq"] = features_df["customer_city"].map(customer_city_frequency)
    features_df["same_state_flag"] = features_df["seller_customer_same_state"]
    features_df["multi_seller_state_flag"] = (features_df["n_seller_states"] > 1).astype(int)
    features_df["multi_category_flag"] = (features_df["n_categories"] > 1).astype(int)
    features_df["avg_product_volume_cm3_safe"] = features_df["avg_product_volume_cm3"]
    features_df["total_product_volume_cm3_safe"] = features_df["total_product_volume_cm3"]
    features_df["estimated_days_per_item"] = np.where(
        features_df["n_items"] > 0,
        features_df["estimated_delivery_days"] / features_df["n_items"],
        np.nan,
    )
    features_df["freight_per_item"] = np.where(
        features_df["n_items"] > 0,
        features_df["total_freight"] / features_df["n_items"],
        np.nan,
    )
    features_df["price_per_kg"] = np.where(
        features_df["avg_product_weight_g"] > 0,
        features_df["avg_item_price"] / (features_df["avg_product_weight_g"] / 1000),
        np.nan,
    )
    return features_df


def build_modeling_dataframe(
    tables: dict[str, pd.DataFrame],
    include_order_id: bool = False,
    include_split_timestamp: bool = False,
) -> pd.DataFrame:
    reviews = deduplicate_reviews(tables["order_reviews"])
    model_df = build_orders_base(tables["orders"], reviews)
    model_df = model_df.drop(
        columns=["order_status", "order_delivered_customer_date", "order_delivered_carrier_date"],
        errors="ignore",
    )

    customer_features = tables["customers"][
        ["customer_id", "customer_zip_code_prefix", "customer_city", "customer_state"]
    ].copy()
    require_unique_key(customer_features, "customer_id", "customers")
    model_df = model_df.merge(customer_features, on="customer_id", how="left", validate="many_to_one")
    if not model_df["order_id"].is_unique:
        raise ValueError("Customer merge violated the order-level grain.")

    items_agg = aggregate_order_items(tables["order_items"])
    model_df = merge_order_level(model_df, items_agg, "order_items_agg")
    model_df = merge_order_level(model_df, aggregate_product_features(tables["order_items"], tables["products"]), "product_agg")
    model_df = merge_order_level(model_df, aggregate_payments(tables["order_payments"], items_agg), "payments_agg")
    model_df = merge_order_level(model_df, aggregate_seller_features(tables["order_items"], tables["sellers"]), "seller_agg")
    model_df["seller_customer_same_state"] = (
        model_df["primary_seller_state"].notna()
        & model_df["customer_state"].notna()
        & (model_df["primary_seller_state"] == model_df["customer_state"])
    ).astype(int)

    features_df = add_engineered_features(model_df)
    from .preprocessing import PRIMARY_FEATURE_COLUMNS

    columns = (
        (["order_id"] if include_order_id else [])
        + (["order_purchase_timestamp"] if include_split_timestamp else [])
        + PRIMARY_FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )
    missing = [column for column in columns if column not in features_df.columns]
    if missing:
        raise ValueError(f"Missing expected engineered feature columns: {missing}")
    output = features_df[columns].copy()
    if include_order_id:
        require_unique_key(output, "order_id", "modeling dataframe")
    return output
