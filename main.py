import pandas as pd
from sklearn.model_selection import train_test_split


credit_card_df = pd.read_csv("./data/UCI_Credit_Card.csv")
print(credit_card_df.head())

X = credit_card_df.drop(columns=["default.payment.next.month"])
y = credit_card_df["default.payment.next.month"]

train_df, test_df = train_test_split(X, y, test_size=0.3, random_state=123)
