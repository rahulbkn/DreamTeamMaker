from typing import Dict, List
import numpy as np
import tensorflow as tf
import sqlite3
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime
import os  # Added missing import

DB_FILE = "player_stats.db"
MODEL_FILE = "player_points_dl_model.keras"
INJURY_MODEL_FILE = "injury_risk_model.keras"
FITNESS_MODEL_FILE = "fitness_model.keras"

conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS feedback (
    player_name TEXT, features TEXT, actual_points REAL, injured INTEGER, fitness_score REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)''')
conn.commit()

def extract_features(player: Dict, pitch: str, weather: str, match_context: Dict) -> tuple:
    static_features = np.array([
        player["avg_points"], player["form_factor"],
        1.1 if pitch == "batting" and player["role"] in ["BAT", "WK"] else 1.0,
        1.05 if weather == "sunny" else 1.0, match_context.get("opponent_strength", 1.0),
        player.get("venue_avg", player["avg_points"]) / player["avg_points"] if player["avg_points"] > 0 else 1.0,
        1.0 - (player.get("matches_played_last_30_days", 0) / 20),
        {"BAT": 1.0, "BOW": 1.2, "ALL": 1.1, "WK": 1.05}.get(player["role"], 1.0),
        1.1 if match_context.get("toss_winner") == player["team"] else 1.0
    ])
    seq_features = np.array(player.get("recent_points", [player["avg_points"]] * 5)[-5:]).reshape(1, 5, 1)
    
    injury_history = json.loads(player.get("injury_history", "[]"))
    injury_count = len(injury_history)
    recent_injury = 1 if any("recovery" in i for i in injury_history[-1:]) else 0
    injury_features = np.array([injury_count, recent_injury, player["matches_played_last_30_days"]])
    
    fitness_history = json.loads(player.get("fitness_history", "[]"))
    days_since_last = (datetime.now() - datetime.strptime(player["last_match_date"], "%Y-%m-%d %H:%M:%S.%f")).days
    fitness_features = np.array([player["matches_played_last_30_days"], days_since_last, injury_count])
    
    return static_features, seq_features, injury_features, fitness_features

def build_model(static_dim: int = 9, timesteps: int = 5):
    static_input = tf.keras.layers.Input(shape=(static_dim,), name="static_input")
    static_x = tf.keras.layers.Dense(128, activation='relu')(static_input)
    static_x = tf.keras.layers.BatchNormalization()(static_x)
    static_x = tf.keras.layers.Dropout(0.3)(static_x)
    static_x = tf.keras.layers.Dense(64, activation='relu')(static_x)
    
    seq_input = tf.keras.layers.Input(shape=(timesteps, 1), name="seq_input")
    seq_x = tf.keras.layers.LSTM(64, return_sequences=True)(seq_input)
    seq_x = tf.keras.layers.LSTM(32)(seq_x)
    
    attention = tf.keras.layers.Attention()([static_x, static_x])
    combined = tf.keras.layers.Concatenate()([attention, seq_x])
    
    x = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(combined)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(1, name="output")(x)
    
    model = tf.keras.Model(inputs=[static_input, seq_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def build_injury_model(input_dim: int = 3):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_fitness_model(input_dim: int = 3):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

scaler = StandardScaler()
injury_scaler = StandardScaler()
fitness_scaler = StandardScaler()
if os.path.exists(MODEL_FILE):
    model = tf.keras.models.load_model(MODEL_FILE)
else:
    model = build_model()
if os.path.exists(INJURY_MODEL_FILE):
    injury_model = tf.keras.models.load_model(INJURY_MODEL_FILE)
else:
    injury_model = build_injury_model()
if os.path.exists(FITNESS_MODEL_FILE):
    fitness_model = tf.keras.models.load_model(FITNESS_MODEL_FILE)
else:
    fitness_model = build_fitness_model()

def predict_points(player: Dict, pitch: str, weather: str, match_context: Dict) -> float:
    static_features, seq_features, _, _ = extract_features(player, pitch, weather, match_context)
    scaled_static = scaler.transform([static_features]) if scaler.is_fitted else static_features.reshape(1, -1)
    return float(model.predict([scaled_static, seq_features], verbose=0)[0][0])

def predict_injury_risk(player: Dict, pitch: str, weather: str, match_context: Dict) -> float:
    _, _, injury_features, _ = extract_features(player, pitch, weather, match_context)
    scaled_injury = injury_scaler.transform([injury_features]) if injury_scaler.is_fitted else injury_features.reshape(1, -1)
    return float(injury_model.predict(scaled_injury, verbose=0)[0][0])

def predict_fitness_score(player: Dict, pitch: str, weather: str, match_context: Dict) -> float:
    _, _, _, fitness_features = extract_features(player, pitch, weather, match_context)
    scaled_fitness = fitness_scaler.transform([fitness_features]) if fitness_scaler.is_fitted else fitness_features.reshape(1, -1)
    return float(fitness_model.predict(scaled_fitness, verbose=0)[0][0]) * 100

def update_model(player_name: str, actual_points: float, static_features: np.ndarray, seq_features: np.ndarray, injured: bool, fitness_score: float):
    feature_str = ','.join(map(str, static_features)) + "|" + ','.join(map(str, seq_features.flatten()))
    c.execute("INSERT INTO feedback (player_name, features, actual_points, injured, fitness_score) VALUES (?, ?, ?, ?, ?)",
              (player_name, feature_str, actual_points, int(injured), fitness_score))
    conn.commit()
    
    c.execute("SELECT features, actual_points, injured, fitness_score FROM feedback")
    data = c.fetchall()
    if len(data) > 10:
        static_X, seq_X, injury_X, fitness_X, points_y, injury_y, fitness_y = [], [], [], [], [], [], []
        from src.data.player_data import load_players
        for row in data:
            features_split = row[0].split("|")
            static = np.array(list(map(float, features_split[0].split(','))))
            seq = np.array(list(map(float, features_split[1].split(',')))).reshape(1, 5, 1)
            p = next(p for p in load_players() if p["name"] == row[0])
            _, _, injury_f, fitness_f = extract_features(p, "batting", "sunny", {"opponent_strength": 1.0})
            static_X.append(static)
            seq_X.append(seq)
            injury_X.append(injury_f)
            fitness_X.append(fitness_f)
            points_y.append(row[1])
            injury_y.append(row[2])
            fitness_y.append(row[3] / 100)
        
        static_X, seq_X, injury_X, fitness_X = np.array(static_X), np.array(seq_X), np.array(injury_X), np.array(fitness_X)
        points_y, injury_y, fitness_y = np.array(points_y), np.array(injury_y), np.array(fitness_y)
        
        scaler.fit(static_X)
        injury_scaler.fit(injury_X)
        fitness_scaler.fit(fitness_X)
        static_X_scaled = scaler.transform(static_X)
        injury_X_scaled = injury_scaler.transform(injury_X)
        fitness_X_scaled = fitness_scaler.transform(fitness_X)
        
        model.fit([static_X_scaled, seq_X], points_y, epochs=10, batch_size=32, verbose=0, validation_split=0.2)
        injury_model.fit(injury_X_scaled, injury_y, epochs=10, batch_size=32, verbose=0, validation_split=0.2)
        fitness_model.fit(fitness_X_scaled, fitness_y, epochs=10, batch_size=32, verbose=0, validation_split=0.2)
        model.save(MODEL_FILE)
        injury_model.save(INJURY_MODEL_FILE)
        fitness_model.save(FITNESS_MODEL_FILE)
