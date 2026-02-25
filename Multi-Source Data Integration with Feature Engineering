import pandas as pd

sales = pd.read_csv("sales.csv")
customers = pd.read_csv("customers.csv")
products = pd.read_csv("products.csv")

data = sales.merge(customers, on="customer_id", how="left")
data = data.merge(products, on="product_id", how="left")

data["order_date"] = pd.to_datetime(data["order_date"])
data["year"] = data["order_date"].dt.year
data["month"] = data["order_date"].dt.month
data["day"] = data["order_date"].dt.day
data["day_of_week"] = data["order_date"].dt.dayofweek

data["total_price"] = data["quantity"] * data["unit_price"]
data["customer_age"] = pd.Timestamp.now().year - data["birth_year"]

category_sales = data.groupby("category")["total_price"].transform("sum")
data["category_sales_ratio"] = data["total_price"] / category_sales

data = pd.get_dummies(data, columns=["gender", "category"], drop_first=True)

data = data.fillna(0)

print(data.head())
