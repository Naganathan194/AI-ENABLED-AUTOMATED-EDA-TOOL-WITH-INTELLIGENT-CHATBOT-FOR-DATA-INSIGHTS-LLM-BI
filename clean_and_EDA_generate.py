import pandas as pd
import numpy as np
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_and_validate_file(uploaded_file, sheet_name=None):
    try:
        file_name = uploaded_file.name.lower()

        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith('.xlsx'):
            excel_file = pd.ExcelFile(uploaded_file)
            if sheet_name is None:
                sheet_name = excel_file.sheet_names[0]
            df = excel_file.parse(sheet_name)
        else:
            logging.error("Unsupported file format. Please upload a CSV or XLSX file.")
            return None
        
        if df.empty:
            logging.error("The file is empty. Please upload a valid dataset.")
            return None
    
        return df
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return None


def clean_data(df):
    """
    Clean the dataframe, but do NOT fill missing values with mean/median/mode.
    Keep NaN as missing. Do not drop columns with high missing % -- just report it.
    Still convert booleans/yes/no, and datetime columns.
    """
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()

        # Track columns that have been converted to avoid double-processing
        converted_to_numeric = set()
        
        # Convert some common categorical/bool columns
        for col in categorical_cols:
            unique_vals = df[col].dropna().astype(str).str.strip().str.lower().unique()
            if set(unique_vals) == set(["yes", "no"]):
                mapping = {"yes": 1, "no": 0}
                df[col] = df[col].astype(str).str.strip().str.lower().map(mapping)
                converted_to_numeric.add(col)
            elif set(unique_vals) == set(["true", "false"]):
                mapping = {"true": 1, "false": 0}
                df[col] = df[col].astype(str).str.strip().str.lower().map(mapping)
                converted_to_numeric.add(col)

        # Try to convert date columns from string
        # Only convert columns that:
        # 1. Haven't been converted to numeric/boolean
        # 2. Have reasonable cardinality (not clearly categorical with few values)
        # 3. Actually contain date-like patterns
        for col in categorical_cols:
            if col in converted_to_numeric:
                continue  # Skip columns we've already converted
            
            # Skip columns with very few unique values that are clearly categorical
            unique_count = df[col].dropna().nunique()
            if unique_count <= 10:
                # Check if it's clearly a categorical column (like smoking_status, recovered)
                sample_values = df[col].dropna().astype(str).str.strip().head(20).tolist()
                # Skip if values are clearly categorical (yes/no, true/false, or short non-date strings)
                is_categorical = False
                for v in sample_values[:min(10, len(sample_values))]:
                    v_lower = str(v).lower()
                    # Check if it's a boolean-like value
                    if v_lower in ['yes', 'no', 'true', 'false', 'y', 'n']:
                        is_categorical = True
                        break
                    # Check if it's a short string without date-like patterns (no digits or date separators)
                    if len(str(v)) < 15 and '/' not in str(v) and '-' not in str(v) and not any(char.isdigit() for char in str(v)[:4]):
                        is_categorical = True
                        break
                
                if is_categorical:
                    continue
            
            try:
                # Try to convert to datetime
                converted = pd.to_datetime(df[col], errors='coerce')
                valid_count = converted.notnull().sum()
                valid_ratio = valid_count / len(df) if len(df) > 0 else 0
                
                # More strict criteria: need high valid ratio AND actual date values in reasonable range
                if valid_ratio > 0.8:
                    # Verify these are actually reasonable dates (not epoch dates from conversion errors)
                    valid_dates = converted.dropna()
                    if len(valid_dates) > 0:
                        min_date = valid_dates.min()
                        max_date = valid_dates.max()
                        # Only convert if dates are in a reasonable range (1900-2100)
                        if min_date.year >= 1900 and max_date.year <= 2100:
                            df[col] = converted
                            logging.info(f"Converted column '{col}' to datetime")
                        else:
                            logging.info(f"Skipping datetime conversion for '{col}': dates out of reasonable range")
            except Exception as e:
                logging.info(f"Column '{col}' could not be converted to datetime: {e}")

        # Do NOT fillna on any column or drop due to missing
        # Optionally, log columns with high missing
        for col in df.columns:
            missing_percentage = df[col].isnull().mean() * 100
            if missing_percentage > 50:
                logging.info(f"Column '{col}' is highly missing: {missing_percentage:.1f}% (but kept)")

        # Remove duplicates only
        df.drop_duplicates(inplace=True)
        df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        return df
    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")
        return None


def enhanced_eda_json(df):
    try:
        eda_summary = {}
        eda_summary["num_rows"] = df.shape[0]
        eda_summary["num_columns"] = df.shape[1]
        columns_info = {}
        for col in df.columns:
            col_info = {}
            col_info["dtype"] = str(df[col].dtype)
            missing_count = int(df[col].isnull().sum())
            missing_percent = round(df[col].isnull().mean() * 100, 2)
            col_info["missing_count"] = missing_count
            col_info["missing_percent"] = missing_percent

            if pd.api.types.is_numeric_dtype(df[col]):
                desc = df[col].describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
                col_info["numeric_stats"] = {
                    "mean": desc.get("mean"),
                    "median": desc.get("50%"),
                    "min": desc.get("min"),
                    "max": desc.get("max"),
                    "std": desc.get("std"),
                    "25%": desc.get("25%"),
                    "75%": desc.get("75%")
                }
                col_info["skewness"] = df[col].skew()
                col_info["kurtosis"] = df[col].kurt()
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count = int(df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0])
                col_info["outlier_count"] = outlier_count
                col_info["outlier_bounds"] = {"lower_bound": lower_bound, "upper_bound": upper_bound}
                # Histogram with "NoData" bin
                counts, bins = np.histogram(df[col].dropna(), bins=10)
                missing_bin = int(df[col].isnull().sum())
                col_info["histogram"] = {
                    "bins": bins.tolist(),
                    "counts": counts.tolist(),
                    "nodata_count": missing_bin
                }
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                # Value counts with NoData bucket
                value_counts = df[col].value_counts(dropna=False)
                
                # Get count of actual missing values (NaN)
                nan_count = df[col].isna().sum()
                
                # Remove string "nan" entries (these are from data conversion, not actual missing values)
                value_counts_clean = value_counts.copy()
                string_nan_keys = [k for k in value_counts_clean.index if isinstance(k, str) and k.lower().strip() == 'nan']
                for key in string_nan_keys:
                    value_counts_clean = value_counts_clean.drop(key)
                
                # Rename actual NaN to "NoData"
                if pd.isna(value_counts_clean.index).any():
                    value_counts_clean = value_counts_clean.rename({np.nan: "NoData"})
                elif nan_count > 0:
                    # If NaN was dropped but we still have missing, add it as "NoData"
                    value_counts_clean["NoData"] = nan_count
                
                vc_dict = value_counts_clean.to_dict()
                # Final check: if there's still a NaN key, rename it to "NoData"
                if any(pd.isna(key) for key in vc_dict.keys()):
                    for k in list(vc_dict.keys()):
                        if pd.isna(k):
                            vc_dict["NoData"] = vc_dict.pop(k)
                col_info["top_categories"] = vc_dict
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_info["min_date"] = str(df[col].min())
                col_info["max_date"] = str(df[col].max())
                try:
                    col_series = pd.to_datetime(df[col], errors='coerce')
                    monthly_counts = col_series.dt.to_period('M').value_counts().sort_index().to_dict()
                    if len(monthly_counts) <= 20:
                        col_info["monthly_distribution"] = {str(k): v for k, v in monthly_counts.items()}
                except Exception as e:
                    logging.info(f"Error generating monthly distribution for column '{col}': {e}")

            columns_info[col] = col_info
        eda_summary["columns"] = columns_info
        eda_summary["missing_data_overall"] = (df.isnull().mean() * 100).round(2).to_dict()
        duplicate_count = int(df.duplicated().sum())
        eda_summary["duplicate_rows"] = duplicate_count
        eda_summary["duplicate_percentage"] = round(duplicate_count / df.shape[0] * 100, 2)
        numeric_df = df.select_dtypes(include=["number"])
        if not numeric_df.empty:
            eda_summary["correlations"] = numeric_df.corr().to_dict()
        return eda_summary
    except Exception as e:
        logging.error(f"Error during EDA summary: {e}")
        return None

