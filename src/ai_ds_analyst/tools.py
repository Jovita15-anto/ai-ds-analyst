import pandas as pd
from langchain_core.tools import tool

def calculate_revenue(dataset_path: str) -> str:
    """Calculate which product generated the highest revenue."""

    df = pd.read_csv(dataset_path)

    df["Revenue"] = df["Quantity"] * df["Price"]

    product_revenue = df.groupby("Product")["Revenue"].sum()

    highest_product = product_revenue.idxmax()
    highest_revenue = product_revenue.max()

    return (
        f"Highest revenue: "
        f"{highest_product} (₹{highest_revenue:,.0f})."
    )


def calculate_quantity(dataset_path: str) -> str:
    """Calculate which product sold the highest quantity."""

    df = pd.read_csv(dataset_path)

    product_quantity = df.groupby("Product")["Quantity"].sum()

    highest_product = product_quantity.idxmax()
    highest_quantity = product_quantity.max()

    return (
        f"Highest quantity sold: "
        f"{highest_product} ({highest_quantity} units)."
    )

def calculate_total_revenue(dataset_path: str) -> str:
    """Calculate the total revenue across all products."""

    df = pd.read_csv(dataset_path)

    df["Revenue"] = df["Quantity"] * df["Price"]

    total_revenue = df["Revenue"].sum()

    return f"Total revenue: ₹{total_revenue:,.0f}."

@tool
def analyze_column(
    dataset_path: str,
    column: str,
    operation: str,
) -> str:
    """Analyze ONE numeric column.

Required arguments:
- dataset_path: path to CSV
- column: exact column name to analyze
- operation: one of average, sum, max, min

Example:
column="price", operation="average"
"""

    df = pd.read_csv(dataset_path)

    # Find the actual column name case-insensitively
    column_map = {col.lower(): col for col in df.columns}

    actual_column = column_map.get(column.lower().strip())

    if actual_column is None:
        return f"Column '{column}' not found in dataset."

    # Check whether the column is numeric
    if not pd.api.types.is_numeric_dtype(df[actual_column]):
        return f"Column '{actual_column}' is not numeric. Please choose a numeric column."

    operation = operation.lower().strip()

    # Normalize common LLM variations
    operation_aliases = {
        "avg": "average",
        "mean": "average",
        "average": "average",

        "total": "sum",
        "sum": "sum",

        "max": "max",
        "maximum": "max",
        "highest": "max",

        "min": "min",
        "minimum": "min",
        "lowest": "min",
    }

    operation = operation_aliases.get(operation, operation)

    if operation == "average":
        value = df[actual_column].mean()

    elif operation == "sum":
        value = df[actual_column].sum()

    elif operation == "max":
        value = df[actual_column].max()

    elif operation == "min":
        value = df[actual_column].min()

    else:
        return f"Unsupported operation: {operation}"

    if actual_column.lower() in ["price", "revenue", "salary", "total spent", "total_spent"]:
        formatted_value = f"₹{value:,.0f}"
    else:
        formatted_value = f"{value:,.0f}"


    # For highest/lowest questions, also return the related name
    if operation in ["max", "min"]:

        if operation == "max":
            row_index = df[actual_column].idxmax()
        else:
            row_index = df[actual_column].idxmin()

        row = df.loc[row_index]

        # Prefer a meaningful name/label column
        preferred_columns = [
            "name",
            "employee",
            "employee_name",
            "customer",
            "customer_name",
            "product",
            "product_name",
        ]

        column_lookup = {
            col.lower().strip(): col
            for col in df.columns
        }

        label_column = None

        for preferred in preferred_columns:
            if preferred in column_lookup:
                label_column = column_lookup[preferred]
                break

        # If no standard name column exists,
        # use the first non-numeric column
        if label_column is None:
            for col in df.columns:
                if col != actual_column and not pd.api.types.is_numeric_dtype(df[col]):
                    label_column = col
                    break

        if label_column:
            label = row[label_column]

            return (
                f"{operation} of {actual_column}: "
                f"{formatted_value}. "
                f"{label_column}: {label}."
            )


    return f"{operation} of {actual_column}: {formatted_value}."

@tool
def dataset_summary(dataset_path: str) -> str:
    """Return a summary of the dataset including shape, columns, data types, and missing values."""

    df = pd.read_csv(dataset_path)

    rows, columns = df.shape

    column_info = []

    for column in df.columns:
        dtype = df[column].dtype
        missing = df[column].isna().sum()

        column_info.append(
            f"{column} ({dtype}) - {missing} missing"
        )

    return (
        f"Dataset contains {rows} rows and {columns} columns.\n\n"
        f"Columns:\n"
        f"{chr(10).join(column_info)}"
    )

@tool
def group_analysis(
    dataset_path: str,
    group_by: str,
    metric: str,
    operation: str,
) -> str:
    """
    Perform grouped analysis on a dataset.

    group_by: column used for grouping.
    metric: numeric column to analyze.
    operation: sum, average, max, or min.
    """

    df = pd.read_csv(dataset_path)

    # Find actual column names case-insensitively
    column_map = {col.lower(): col for col in df.columns}

    # Normalize metric aliases (so "total_revenue", "sales", "income" etc. all map to "revenue")
    metric_aliases = {
        "revenue": "revenue",
        "sales": "revenue",
        "income": "revenue",
        "total_revenue": "revenue",
        "total revenue": "revenue",

        "price": "price",
        "prices": "price",

        "quantity": "quantity",
        "quantities": "quantity",
        "units": "quantity",
        "units_sold": "quantity",
    }

    metric_key = metric_aliases.get(metric.lower().strip(), metric.lower().strip())

    actual_group_by = column_map.get(group_by.lower().strip())
    actual_metric = column_map.get(metric_key)

    if actual_group_by is None:
        return f"Column '{group_by}' not found in dataset."

    if actual_metric is None:
        # Revenue may be a calculated metric
        if metric_key == "revenue":
            df["Revenue"] = df["Quantity"] * df["Price"]
            actual_metric = "Revenue"
        else:
            return f"Column '{metric}' not found in dataset."

    # Normalize operation
    operation_aliases = {
        "total": "sum",
        "sum": "sum",

        "avg": "average",
        "mean": "average",
        "average": "average",

        "max": "max",
        "maximum": "max",
        "highest": "max",

        "min": "min",
        "minimum": "min",
        "lowest": "min",
    }

    operation = operation_aliases.get(
        operation.lower().strip(),
        operation.lower().strip()
    )

    if operation == "sum":
        result = df.groupby(actual_group_by)[actual_metric].sum()

    elif operation == "average":
        result = df.groupby(actual_group_by)[actual_metric].mean()

    elif operation == "max":
        result = df.groupby(actual_group_by)[actual_metric].max()

    elif operation == "min":
        result = df.groupby(actual_group_by)[actual_metric].min()

    else:
        return f"Unsupported operation: {operation}"

    # Format output
    lines = []

    for group, value in result.items():

        if actual_metric.lower() in ["price", "revenue"]:
            formatted_value = f"₹{value:,.0f}"
        else:
            formatted_value = f"{value:,.0f}"

        lines.append(
            f"{group}: {formatted_value}"
        )

    return (
        f"{operation} of {actual_metric} by "
        f"{actual_group_by}:\n"
        + "\n".join(lines)
    )

@tool
def product_analysis(
    dataset_path: str,
    metric: str,
    operation: str
) -> str:
    """Analyze products using a selected metric and operation."""

    df = pd.read_csv(dataset_path)

    metric = metric.lower().strip()
    operation = operation.lower().strip()

    # Normalize common LLM variations
    metric_aliases = {
        "price": "price",
        "prices": "price",

        "quantity": "quantity",
        "quantities": "quantity",
        "units": "quantity",
        "units_sold": "quantity",
        "sold_units": "quantity",

        "revenue": "revenue",
        "sales": "revenue",
        "income": "revenue",
    }

    operation_aliases = {
        "highest": "highest",
        "high": "highest",
        "maximum": "highest",
        "max": "highest",
        "most": "highest",

        "lowest": "lowest",
        "low": "lowest",
        "minimum": "lowest",
        "min": "lowest",
        "least": "lowest",
    }

    metric = metric_aliases.get(metric, metric)
    operation = operation_aliases.get(operation, operation)

    metric_map = {
        "price": "Price",
        "quantity": "Quantity",
        "revenue": "Revenue",
    }

    if metric not in metric_map:
        return f"Unsupported metric: {metric}"

    metric_column = metric_map[metric]

    if metric_column == "Revenue":
        df["Revenue"] = df["Quantity"] * df["Price"]

    if metric == "revenue":
        grouped = df.groupby("Product")["Revenue"]

        if operation == "highest":
            result = grouped.sum()
            product = result.idxmax()
            value = result.max()

        elif operation == "lowest":
            result = grouped.sum()
            product = result.idxmin()
            value = result.min()

        else:
            return f"Unsupported operation for revenue: {operation}"

    elif metric == "quantity":
        grouped = df.groupby("Product")["Quantity"]

        if operation == "highest":
            result = grouped.sum()
            product = result.idxmax()
            value = result.max()

        elif operation == "lowest":
            result = grouped.sum()
            product = result.idxmin()
            value = result.min()

        else:
            return f"Unsupported operation for quantity: {operation}"

    elif metric == "price":
        grouped = df.groupby("Product")["Price"]

        if operation == "highest":
            result = grouped.mean()
            product = result.idxmax()
            value = result.max()

        elif operation == "lowest":
            result = grouped.mean()
            product = result.idxmin()
            value = result.min()

        else:
            return f"Unsupported operation for price: {operation}"

    if metric == "revenue":
        formatted_value = f"₹{value:,.0f}"

    elif metric == "price":
        formatted_value = f"₹{value:,.0f}"

    else:
        formatted_value = f"{value:,.0f}"

    return (
        f"Product with {operation} {metric}: "
        f"{product} ({formatted_value})."
    )

@tool
def dataset_insights(dataset_path: str) -> str:
    """Generate important business insights from the dataset."""

    df = pd.read_csv(dataset_path)

    insights = []

    # Prefer a "name"-like column over an ID-like column
    # Use "not numeric" instead of "== object" so this works
    # across pandas versions (some report text as 'str', not 'object')
    label_column = None

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and "id" not in col.lower():
            label_column = col
            break

    if label_column is None:
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                label_column = col
                break

    numeric_columns = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    if not numeric_columns:
        return "No numeric columns were found to generate insights from."

    for col in numeric_columns:
        highest_row = df.loc[df[col].idxmax()]
        lowest_row = df.loc[df[col].idxmin()]

        highest_label = highest_row[label_column] if label_column else f"row {highest_row.name}"
        lowest_label = lowest_row[label_column] if label_column else f"row {lowest_row.name}"

        if col.lower() in ["price", "revenue", "total spent", "total_spent", "totalspent"]:
            insights.append(f"Highest {col}: {highest_label} (₹{highest_row[col]:,.0f}).")
            insights.append(f"Lowest {col}: {lowest_label} (₹{lowest_row[col]:,.0f}).")
        else:
            insights.append(f"Highest {col}: {highest_label} ({highest_row[col]:,.0f}).")
            insights.append(f"Lowest {col}: {lowest_label} ({lowest_row[col]:,.0f}).")

    return "\n".join(insights)

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv(r"C:\Users\kavij\Desktop\AI DS Analyst\data\customers.csv")
    print("COLUMNS:", list(df.columns))
    print("DTYPES:\n", df.dtypes)

    result = dataset_insights.invoke({
        "dataset_path": r"C:\Users\kavij\Desktop\AI DS Analyst\data\customers.csv"
    })
    print(result)