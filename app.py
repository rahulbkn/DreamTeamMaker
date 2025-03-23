from flask import Flask, render_template, request, redirect, url_for, flash
from src.data.player_data import load_players, sync_player_data
from src.models.optimizer import generate_team, select_leaders, get_differential_pick
from src.models.predictor import predict_points, predict_injury_risk, predict_fitness_score, update_model, extract_features
from src.utils.validators import is_valid_team
from config.settings import MAX_CREDITS, ROLE_LIMITS, TEAM_SIZE
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'fallback-secret-key')

players = load_players()
current_team = []

@app.route('/')
def home():
    return render_template('home.html', max_credits=MAX_CREDITS, team_size=TEAM_SIZE, players=players)

@app.route('/generate_team', methods=['GET', 'POST'])
def generate_team_route():
    global current_team
    if request.method == 'POST':
        pitch = request.form['pitch']
        weather = request.form['weather']
        risk_level = request.form['risk_level']
        must_include = request.form.getlist('must_include')
        match_id = request.form.get('match_id', None)
        opponent_team = request.form.get('opponent_team', 'Unknown')
        toss_winner = request.form.get('toss_winner', 'Unknown')
        city = request.form.get('city', 'Unknown')
        
        try:
            global players
            match_context = {"opponent_strength": 1.0, "toss_winner": toss_winner}
            players = sync_player_data(players, pitch, weather, match_id, city)
            current_team = generate_team(players, pitch, weather, must_include, risk_level)
            if not is_valid_team(current_team):
                flash("Generated team doesn't meet all constraints.", "error")
            return redirect(url_for('view_team', pitch=pitch, weather=weather, match_id=match_id,
                                   opponent_team=opponent_team, toss_winner=toss_winner, city=city))
        except Exception as e:
            flash(f"Error generating team: {str(e)}", "error")
    return render_template('home.html', players=players)

@app.route('/team')
def view_team():
    if not current_team:
        flash("No team generated yet. Please generate a team first.", "error")
        return redirect(url_for('home'))
    
    pitch = request.args.get('pitch', 'batting')
    weather = request.args.get('weather', 'sunny')
    match_id = request.args.get('match_id', None)
    opponent_team = request.args.get('opponent_team', 'Unknown')
    toss_winner = request.args.get('toss_winner', 'Unknown')
    city = request.args.get('city', 'Unknown')
    match_context = {"opponent_strength": 1.0, "toss_winner": toss_winner}
    
    try:
        total_credits = sum(p["credits"] for p in current_team)
        total_points = sum(predict_points(p, pitch, weather, match_context) for p in current_team)
        captain, vice_captain = select_leaders(current_team, pitch, weather)
        captain_points = predict_points(captain, pitch, weather, match_context)
        vice_captain_points = predict_points(vice_captain, pitch, weather, match_context)
        adjusted_points = (total_points - captain_points - vice_captain_points + 2 * captain_points + 1.5 * vice_captain_points)
        
        differential = get_differential_pick(current_team, players, pitch, weather)
        differential_points = predict_points(differential, pitch, weather, match_context) if differential else None
        
        team_with_details = [
            {
                'role': p['role'], 'name': p['name'], 'team': p['team'], 'credits': p['credits'],
                'ownership': p['ownership'], 'predicted_points': predict_points(p, pitch, weather, match_context),
                'injury_risk': predict_injury_risk(p, pitch, weather, match_context) * 100,
                'injury_status': p['injury_status'], 'fitness_score': predict_fitness_score(p, pitch, weather, match_context)
            } for p in current_team
        ]
        
        return render_template('team.html', team=team_with_details, total_credits=total_credits,
                              total_points=total_points, captain=captain, vice_captain=vice_captain,
                              captain_points=captain_points, vice_captain_points=vice_captain_points,
                              adjusted_points=adjusted_points, differential=differential,
                              differential_points=differential_points, pitch=pitch, weather=weather,
                              match_id=match_id, opponent_team=opponent_team, toss_winner=toss_winner, city=city)
    except Exception as e:
        flash(f"Error displaying team: {str(e)}", "error")
        return redirect(url_for('home'))

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    global current_team
    if not current_team:
        flash("No team to provide feedback for.", "error")
        return redirect(url_for('home'))
    
    pitch = request.args.get('pitch', 'batting')
    weather = request.args.get('weather', 'sunny')
    match_id = request.args.get('match_id', None)
    opponent_team = request.args.get('opponent_team', 'Unknown')
    toss_winner = request.args.get('toss_winner', 'Unknown')
    city = request.args.get('city', 'Unknown')
    match_context = {"opponent_strength": 1.0, "toss_winner": toss_winner}
    
    if request.method == 'POST':
        try:
            for player in current_team:
                actual_points = float(request.form.get(f"points_{player['name']}", 0))
                injured = bool(request.form.get(f"injured_{player['name']}", False))  # Fixed f-string syntax
                fitness_score = float(request.form.get(f"fitness_{player['name']}", player["fitness_score"]))
                static_features, seq_features, _, _ = extract_features(player, pitch, weather, match_context)
                update_model(player["name"], actual_points, static_features, seq_features, injured, fitness_score)
            flash("Feedback recorded and model updated!", "success")
            return redirect(url_for('view_team', pitch=pitch, weather=weather, match_id=match_id,
                                   opponent_team=opponent_team, toss_winner=toss_winner, city=city))
        except Exception as e:
            flash(f"Invalid feedback data: {str(e)}", "error")
    
    return render_template('feedback.html', team=current_team, pitch=pitch, weather=weather,
                          match_id=match_id, opponent_team=opponent_team, toss_winner=toss_winner, city=city)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
