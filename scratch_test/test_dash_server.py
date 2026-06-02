import os

import dash
import dash_bootstrap_components as dbc
from dash import html

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.layout = html.Div([html.H1("Simple Dash Test"), html.P("If you can see this, Dash port binding works!")])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"Starting test server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
