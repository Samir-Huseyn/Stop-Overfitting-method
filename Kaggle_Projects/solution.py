import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("NetFlix.csv")
print(df.info())
print(df.head(3))

df = df.drop(columns=["title", "duration", "date_added", "show_id", "description", "director", "cast", "country", "release_year", "rating"])
df["genres"] = df["genres"].str.replace("Movies", "").str.replace("TV Shows", "").str.replace("Series", "")
new_col = df["genres"].str.get_dummies(sep=",")
df = pd.concat([df, new_col], axis=1)
df = df.drop(columns = ["genres"])
df["type"] = df["type"].map({"Movie": 0, "TV Show": 1})
print(df.shape)
print(df.head())


X = df.drop(columns = ["type"]).values
y = df["type"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

print(X_train.shape)

model = Sequential([
    Dense(units=16, activation="relu", input_shape = (67,)),
    Dropout(0.2),
    Dense(units=8, activation="relu"),
    Dropout(0.1),
    Dense(units=6, activation="relu"),
    Dense(units=1, activation="sigmoid")
])

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"] )
model.fit(X_train, y_train, epochs = 100, batch_size = 10, validation_data = (X_test, y_test), callbacks = [early_stop])

loss, accuracy = model.evaluate(X_test, y_test)
print(loss)
print(accuracy)